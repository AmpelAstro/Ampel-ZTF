
import pytest
import asyncio
import time
import os

from ampel.ztf.t0.ZTFAlertStreamController import ZTFAlertStreamController
from ampel.model.ZTFLegacyChannelTemplate import ZTFLegacyChannelTemplate
from ampel.config.builder.FirstPassConfig import FirstPassConfig
from ampel.model.ProcessModel import ProcessModel
from ampel.log.AmpelLogger import AmpelLogger
from ampel.config.AmpelConfig import AmpelConfig
from ampel.util import concurrent

def t0_process(kwargs):
    first_pass_config = FirstPassConfig()
    first_pass_config['resource']['ampel-ztf/kafka'] = {"group": "nonesuch"}
    return ProcessModel(**ZTFLegacyChannelTemplate(**kwargs).get_processes(AmpelLogger.get_logger(), first_pass_config)[0])

def make_controller(config, processes, klass=ZTFAlertStreamController):
    return klass(
        config=config,
        priority="standard",
        source={"stream": "ztf_uw_public", **config.get('resource.ampel-ztf/kafka')},
        processes=processes,
        )

@pytest.fixture
def first_pass_config():
    return {
        "resource": {
            "ampel-ztf/kafka": {
                "broker": "nonesuch:9092",
                "group": "nonesuch",
            }
        }
    }
    first_pass_config = FirstPassConfig()
    first_pass_config['resource']['ampel-ztf/kafka'] = {
        "broker": "nonesuch:9092",
        "group": "nonesuch",
    }
    return first_pass_config

@pytest.fixture
def config(first_pass_config):
    return AmpelConfig(dict(first_pass_config))

def test_merge_processes(config):
    # matching processes
    processes = [t0_process({
        "channel": name,
        "auto_complete": False,
        "template": "ztf_uw_public",
        "t0_filter": {"unit": "NoFilter"}
    }) for name in ("foo", "bar")]
    len(ZTFAlertStreamController.merge_processes(processes).processor.config['directives']) == 2

    # incompatible processor configs
    processes[0].processor.config["iter_max"] = 100
    processes[1].processor.config["iter_max"] = 200
    with pytest.raises(AssertionError):
        make_controller(config, processes)

    # incompatible controller configs
    processes = [t0_process({
        "channel": name,
        "auto_complete": False,
        "template": stream,
        "t0_filter": {"unit": "NoFilter"}
    }) for name, stream in zip(("foo", "bar"),("ztf_uw_private", "ztf_uw_public"))]
    with pytest.raises(AssertionError):
        make_controller(config, processes)

class PotemkinZTFAlertStreamController(ZTFAlertStreamController):
    @staticmethod
    @concurrent.process
    def run_mp_process(
        config,
        secrets,
        p,
        source,
        log_profile: str = "default",
    ) -> bool:
        print(f"{os.getpid()} is sleepy...")
        time.sleep(1)
        print(f"{os.getpid()} is done!")
        return True

def test_scale(config):
    processes = [t0_process({
        "channel": name,
        "active": True,
        "auto_complete": False,
        "template": "ztf_uw_public",
        "t0_filter": {"unit": "NoFilter"}
    }) for name in ("foo", "bar")]
    controller = make_controller(config, processes, PotemkinZTFAlertStreamController)
    assert controller.processes[0].active
    
    async def main():
        try:
            r = asyncio.create_task(controller.run())
            await asyncio.sleep(0.5)
            controller.scale(multiplier=2)
            await asyncio.sleep(2)
            controller.scale(multiplier=3)
            await asyncio.sleep(2)
            controller.scale(multiplier=1)
            await asyncio.sleep(1)
        finally:
            controller.stop()
        await r
    asyncio.run(main())
