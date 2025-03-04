import asyncio
import os
import time
from contextlib import suppress

import pytest
import requests

from ampel.abstract.AbsEventUnit import AbsEventUnit
from ampel.config.AmpelConfig import AmpelConfig
from ampel.core.EventHandler import EventHandler
from ampel.dev.DevAmpelContext import DevAmpelContext
from ampel.log.AmpelLogger import AmpelLogger
from ampel.metrics.AmpelMetricsRegistry import AmpelMetricsRegistry
from ampel.model.ProcessModel import ProcessModel
from ampel.secret.AmpelVault import AmpelVault
from ampel.secret.DictSecretProvider import DictSecretProvider
from ampel.template.ZTFLegacyChannelTemplate import ZTFLegacyChannelTemplate
from ampel.util import concurrent
from ampel.util.template import apply_templates
from ampel.ztf.t0.load.ZTFArchiveAlertLoader import ZTFArchiveAlertLoader
from ampel.ztf.t0.ZTFAlertStreamController import ZTFAlertStreamController


def t0_process(kwargs, first_pass_config):
    first_pass_config["resource"]["ampel-ztf/kafka"] = {
        "group": "nonesuch",
        "broker": "nonesuch:9092",
    }
    # explicitly add version, which would otherwise be synthesized by ProcessMorpher
    template = ZTFLegacyChannelTemplate(**kwargs)
    return ProcessModel(
        **(
            template.get_processes(AmpelLogger.get_logger(), first_pass_config)[0]
            | {"version": template.version}
        )
    )


def make_controller(config, processes, klass=ZTFAlertStreamController):
    return klass(
        config=config,
        priority="standard",
        processes=processes,
    )


@pytest.fixture
def config(first_pass_config):
    first_pass_config["resource"]["ampel-ztf/kafka"] = {
        "broker": "nonesuch:9092",
        "group": "nonesuch",
    }
    return AmpelConfig(dict(first_pass_config))


def test_merge_processes(config, first_pass_config):
    # matching processes
    processes = [
        t0_process(
            {
                "channel": name,
                "version": 0,
                "auto_complete": False,
                "template": "ztf_uw_public",
                "t0_filter": {"unit": "NoFilter"},
            },
            first_pass_config,
        )
        for name in ("foo", "bar")
    ]
    for pm in processes:
        pm.controller.config = {"priority": pm.name.split("|")[-1]}
    assert (
        len(
            ZTFAlertStreamController.merge_processes(processes).processor.config[
                "directives"
            ]
        )
        == 2
    )

    # incompatible processor configs
    processes[0].processor.config["iter_max"] = 100
    processes[1].processor.config["iter_max"] = 200
    with pytest.raises(AssertionError):
        make_controller(config, processes)

    # incompatible controller configs
    processes = [
        t0_process(
            {
                "channel": name,
                "version": 0,
                "auto_complete": False,
                "template": stream,
                "t0_filter": {"unit": "NoFilter"},
            },
            first_pass_config,
        )
        for name, stream in zip(
            ("foo", "bar"), ("ztf_uw_private", "ztf_uw_public"), strict=False
        )
    ]
    for pm in processes:
        pm.controller.config = {"priority": pm.name.split("|")[-1]}
    with pytest.raises(AssertionError):
        make_controller(config, processes)


class PotemkinZTFAlertStreamController(ZTFAlertStreamController):
    @staticmethod
    @concurrent.process
    def run_mp_process(
        config,
        secrets,
        p,
        log_profile: str = "default",
    ) -> bool:
        print(f"{os.getpid()} is sleepy...")
        time.sleep(1)
        print(f"{os.getpid()} is done!")
        return True


@pytest.fixture
def potemkin_controller(config, first_pass_config):
    processes = [
        t0_process(
            {
                "channel": name,
                "version": 0,
                "active": True,
                "auto_complete": False,
                "template": "ztf_uw_public",
                "t0_filter": {"unit": "NoFilter"},
            },
            first_pass_config,
        )
        for name in ("foo", "bar")
    ]
    controller = make_controller(config, processes, PotemkinZTFAlertStreamController)
    assert controller.processes[0].active
    return controller


@pytest.mark.asyncio
async def test_process_gauge(potemkin_controller):
    def process_count():
        return AmpelMetricsRegistry.registry().get_sample_value(
            "ampel_processes",
            {"tier": "0", "process": potemkin_controller._process.name},  # noqa: SLF001
        )

    r = asyncio.create_task(potemkin_controller.run())
    with suppress(asyncio.exceptions.TimeoutError):
        await asyncio.wait_for(asyncio.shield(r), 0.5)
    try:
        assert process_count() == 1
    finally:
        potemkin_controller.stop()
        await r
    assert process_count() == 0


@pytest.mark.asyncio
async def test_scale(potemkin_controller):
    def process_count():
        return AmpelMetricsRegistry.registry().get_sample_value(
            "ampel_processes",
            {"tier": "0", "process": potemkin_controller._process.name},  # noqa: SLF001
        )

    try:
        r = asyncio.create_task(potemkin_controller.run())
        await asyncio.sleep(0.5)
        assert process_count() == 1
        potemkin_controller.scale(multiplier=2)
        await asyncio.sleep(2)
        assert process_count() == 2
        potemkin_controller.scale(multiplier=3)
        await asyncio.sleep(2)
        assert process_count() == 3
        potemkin_controller.scale(multiplier=1)
        await asyncio.sleep(1)
        assert process_count() == 1
    finally:
        potemkin_controller.stop()
    await r


@pytest.mark.asyncio
async def test_stop(potemkin_controller):
    r = asyncio.create_task(potemkin_controller.run())
    with suppress(asyncio.exceptions.TimeoutError):
        await asyncio.wait_for(asyncio.shield(r), 0.5)
    potemkin_controller.stop()
    assert isinstance((await r)[0], asyncio.CancelledError)


@pytest.fixture
def archive_api_token() -> str:
    try:
        return os.environ["ARCHIVE_TOKEN"]
    except KeyError:
        pytest.skip("archive test requires api token")


@pytest.fixture
def topic_stream_token() -> str:
    if not (token := os.environ.get("ARCHIVE_TOPIC_TOKEN")):
        pytest.skip("archive test requires stream token")
    base_url = "https://ampel.zeuthen.desy.de/api/ztf/archive/v3"
    step = 10000
    chunk = 5
    with requests.Session() as session:
        response = session.get(f"{base_url}/topic/{token}")
        response.raise_for_status()
        size = response.json()["size"]

        response = session.post(
            f"{base_url}/streams/from_topic",
            json={"topic": token, "chunk_size": chunk, "step": step},
        )
        response.raise_for_status()
        assert response.json()["remaining"]["items"] == (size + step - 1) // step
        return response.json()["resume_token"]


def test_archive_source(topic_stream_token: str):
    assert isinstance(topic_stream_token, str)
    source = ZTFArchiveAlertLoader(stream=topic_stream_token)
    alerts = list(iter(source))
    assert len(alerts) == 8


def test_archive_source_for_objectid(archive_api_token):
    source = ZTFArchiveAlertLoader(
        stream={
            "ztf_name": "ZTF21abbxdcm",
            "with_history": False,
            "archive_token": archive_api_token,
            "jd_start": 2459352,
            "jd_end": 2459366,
            "programid": 1,
        }
    )
    alerts = list(iter(source))
    assert len(alerts) == 10


def test_T4ZTFArchiveTokenGenerator(mock_context: DevAmpelContext, archive_api_token):
    mock_context.loader.vault = AmpelVault(
        [DictSecretProvider({"archive_token": archive_api_token})]
    )

    t4: AbsEventUnit = mock_context.new_context_unit(  # type: ignore[assignment]
        "T4Processor",
        process_name="foo",
        raise_exc=True,
        execute=[
            {
                "unit": "T4RunTimeContextUpdater",
                "config": {
                    "execute": [
                        {
                            "unit": "T4ZTFArchiveTokenGenerator",
                            "config": {
                                "resource_name": "%%ztf_stream_token",
                                "archive_token": {"label": "archive_token"},
                            },
                        }
                    ]
                },
            }
        ],
    )

    event_handler = EventHandler(t4.process_name, mock_context.get_database())
    event_handler.register(run_id=mock_context.new_run_id(), tier=4)
    t4.proceed(event_handler)
    assert "%%ztf_stream_token" in mock_context.run_time_aliases

    unit_conf = {"stream": "%%ztf_stream_token"}
    apply_templates(
        mock_context, "resolve_run_time_aliases", unit_conf, AmpelLogger.get_logger()
    )
    source = ZTFArchiveAlertLoader(**unit_conf)
    alerts = list(iter(source))
    assert len(alerts) == 27
