import itertools
import os
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager

import fastavro
import pytest

from ampel.abstract.AbsIngester import AbsIngester
from ampel.core.AmpelContext import AmpelContext
from ampel.dev.UnitTestAlertSupplier import UnitTestAlertSupplier
from ampel.ingest.ChainedIngestionHandler import ChainedIngestionHandler
from ampel.ingest.T0Compiler import T0Compiler
from ampel.log.AmpelLogger import DEBUG, AmpelLogger
from ampel.model.ingest.IngestDirective import IngestDirective
from ampel.model.UnitModel import UnitModel
from ampel.mongo.update.DBUpdatesBuffer import DBUpdatesBuffer
from ampel.mongo.update.MongoT0Ingester import MongoT0Ingester
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.secret.AmpelVault import AmpelVault
from ampel.secret.DictSecretProvider import DictSecretProvider
from ampel.ztf.alert.ZiAlertSupplier import ZiAlertSupplier
from ampel.ztf.ingest.ZiArchiveMuxer import ZiArchiveMuxer
from ampel.ztf.ingest.ZiCompilerOptions import ZiCompilerOptions
from ampel.ztf.ingest.ZiDataPointShaper import ZiDataPointShaperBase


def _make_muxer(context: AmpelContext, model: UnitModel) -> ZiArchiveMuxer:
    run_id = 0
    logger = AmpelLogger.get_logger()

    return context.loader.new_context_unit(
        model=model,
        sub_type=ZiArchiveMuxer,
        context=context,
        logger=logger,
    )


def get_supplier(loader):
    return ZiAlertSupplier(
        deserialize="avro", loader=UnitTestAlertSupplier(alerts=list(loader))
    )


@pytest.fixture
def raw_alert_dicts(avro_packets):
    def gen():
        for f in avro_packets():
            yield next(fastavro.reader(f))

    return gen


@pytest.fixture
def alerts(raw_alert_dicts):
    def gen():
        for d in raw_alert_dicts():
            yield ZiAlertSupplier.shape_alert_dict(d)

    return gen


@pytest.fixture
def superseded_alerts(superseded_packets):
    def gen():
        for f in superseded_packets():
            yield ZiAlertSupplier.shape_alert_dict(next(fastavro.reader(f)))

    return gen


@pytest.fixture
def consolidated_alert(raw_alert_dicts):
    """
    Make one mega-alert containing all photopoints for an object, similar to
    the one returned by ArchiveDB.get_photopoints_for_object
    """
    candidates = []
    prv_candidates = []
    upper_limits = []
    for alert in itertools.islice(raw_alert_dicts(), 0, 1):
        oid = alert["objectId"]
        candidates.append((oid, alert["candidate"]))
        for prv in alert["prv_candidates"]:
            if prv.get("magpsf") is None:
                upper_limits.append((oid, prv))
            else:
                prv_candidates.append((oid, prv))
    # ensure exactly one observation per jd. in case of conflicts, sort by
    # candidate > prv_candidate > upper_limit, then pid
    photopoints = defaultdict(dict)
    for row in ([upper_limits], [prv_candidates], [candidates]):
        for pp in sorted(row[0], key=lambda pp: (pp[0], pp[1]["jd"], pp[1]["pid"])):
            photopoints[pp[0]][pp[1]["jd"]] = pp[1]
    assert len(photopoints) == 1
    objectId = next(iter(photopoints.keys()))
    datapoints = sorted(
        photopoints[objectId].values(), key=lambda pp: pp["jd"], reverse=True
    )
    candidate = datapoints.pop(0)
    return {
        "objectId": objectId,
        "candid": candidate["candid"],
        "programid": candidate["programid"],
        "candidate": candidate,
        "prv_candidates": datapoints,
    }


@pytest.mark.parametrize(
    "model",
    [
        UnitModel(unit="ZiArchiveMuxer", config={"history_days": 30}),
    ],
)
def test_instantiate(mock_context: AmpelContext, model):
    _make_muxer(mock_context, model)


@pytest.fixture
def _mock_get_photopoints(mocker, consolidated_alert):
    # mock get_photopoints to return first alert
    mocker.patch(
        "ampel.ztf.ingest.ZiArchiveMuxer.ZiArchiveMuxer.get_photopoints",
        return_value=consolidated_alert,
    )


@pytest.fixture
def mock_archive_muxer(dev_context, _mock_get_photopoints):
    return _make_muxer(
        dev_context, UnitModel(unit="ZiArchiveMuxer", config={"history_days": 30})
    )


@pytest.fixture
def t0_ingester(dev_context):
    run_id = 0
    logger = AmpelLogger.get_logger()
    updates_buffer = DBUpdatesBuffer(dev_context.db, run_id=run_id, logger=logger)
    ingester = MongoT0Ingester(updates_buffer=updates_buffer)
    compiler = T0Compiler(tier=0, run_id=run_id)
    return ingester, compiler


def test_get_earliest_jd(
    t0_ingester: tuple[MongoT0Ingester, T0Compiler], mock_archive_muxer, alerts
):
    """earliest jd is stable under out-of-order ingestion"""

    alert_list = list(alerts())

    ingester, compiler = t0_ingester

    for i in [2, 0, 1]:
        datapoints = ZiDataPointShaperBase().process(
            alert_list[i].datapoints, stock=alert_list[i].stock
        )
        compiler.add(datapoints, channel="EXAMPLE_TNS_MSIP", ttl=None, trace_id=0)
        compiler.commit(ingester, 0)

        assert mock_archive_muxer.get_earliest_jd(
            alert_list[i].stock, datapoints
        ) == min(
            dp["body"]["jd"] for dp in [el for el in datapoints if el["id"] > 0]
        ), "min jd is min jd of last ingested alert"


@contextmanager
def get_handler(context, directives, run_id=0) -> Generator[ChainedIngestionHandler, None, None]:
    logger = AmpelLogger.get_logger(console={"level": DEBUG})
    with context.loader.new_context_unit(
        UnitModel(unit="MongoIngester"),
        context=context,
        logger=logger,
        run_id=run_id,
        tier=0,
        process_name="test_ingestion_handler",
        sub_type=AbsIngester,
    ) as ingester:
        yield ChainedIngestionHandler(
            context=context,
            logger=logger,
            run_id=run_id,
            ingester=ingester,
            directives=directives,
            compiler_opts=ZiCompilerOptions(),
            shaper=UnitModel(unit="ZiDataPointShaper"),
            trace_id={},
            tier=0,
        )


@pytest.mark.usefixtures("_mock_get_photopoints")
def test_integration(mock_context, alerts):
    mock_context.add_channel("EXAMPLE_TNS_MSIP", ["ZTF", "ZTF_PUB", "ZTF_PRIV"])
    directive = {
        "channel": "EXAMPLE_TNS_MSIP",
        "ingest": {
            "combine": [
                {"unit": "ZiT1Combiner", "state_t2": [{"unit": "DemoLightCurveT2Unit"}]}
            ],
            "mux": {
                "unit": "ZiArchiveMuxer",
                "config": {"history_days": 30},
                "combine": [
                    {
                        "unit": "ZiT1Combiner",
                        "state_t2": [{"unit": "DemoLightCurveT2Unit"}],
                    }
                ],
            },
        },
    }

    with get_handler(mock_context, [IngestDirective(**directive)]) as handler:

        stock = mock_context.db.get_collection("stock")
        t0 = mock_context.db.get_collection("t0")
        t1 = mock_context.db.get_collection("t1")
        t2 = mock_context.db.get_collection("t2")
        assert t0.count_documents({}) == 0

        alert_list = list(alerts())

        handler.ingest(
            alert_list[1].datapoints,
            stock_id=alert_list[1].stock,
            filter_results=[(0, True)],
        )

    ZiArchiveMuxer.get_photopoints.assert_called_once()

    # note lack of handler.updates_buffer.push_updates() here;
    # ZiAlertContentIngester has to be synchronous to deal with superseded
    # photopoints

    assert stock.count_documents({}) == 1
    doc = next(stock.find())
    assert doc["tag"] == ["ZTF"]
    assert doc["name"] == ["ZTF18abxhyqv"]

    assert t0.count_documents({}) == len(alert_list[1].datapoints) + len(
        alert_list[0].datapoints
    ), "datapoints ingested for archival alert"

    assert t1.count_documents({}) == 2, "two compounds produced"
    assert t2.count_documents({}) == 2, "two t2 docs produced"

    assert t2.find_one(
        {"link": t1.find_one({"dps": {"$size": len(alert_list[1].datapoints)}})["link"]}
    )
    assert t2.find_one(
        {
            "link": t1.find_one(
                {
                    "dps": {
                        "$size": len(alert_list[1].datapoints)
                        + len(alert_list[0].datapoints)
                    }
                }
            )["link"]
        }
    )


@pytest.fixture
def archive_token(mock_context, monkeypatch):
    if not (token := os.environ.get("ARCHIVE_TOKEN")):
        pytest.skip("archive test requires token")
    monkeypatch.setattr(
        mock_context.loader,
        "vault",
        AmpelVault(
            [
                DictSecretProvider({"ztf/archive/token": token}),
                *mock_context.loader.vault.providers,
            ]
        ),
    )
    return token


def test_get_photopoints_from_api(mock_context, archive_token):
    """
    ZiT1ArchivalCompoundIngester can communicate with the archive service
    """
    muxer = _make_muxer(
        mock_context, UnitModel(unit="ZiArchiveMuxer", config={"history_days": 30})
    )
    alert_pre = muxer.get_photopoints(
        "ZTF18abcfcoo", jd_center=2458300, time_pre=30, time_post=0
    )

    alert_post = muxer.get_photopoints(
        "ZTF18abcfcoo", jd_center=2458270, time_pre=0, time_post=30
    )

    assert len(alert_pre["prv_candidates"]) == 10
    assert len(alert_post["prv_candidates"]) == 10


def test_deduplication(
    dev_context, t0_ingester: tuple[MongoT0Ingester, T0Compiler], alerts
):
    """
    Database gets only one copy of each datapoint
    """

    alert_list = list(itertools.islice(alerts(), 1, None))

    ingester, compiler = t0_ingester
    filter_pps = [{"attribute": "candid", "operator": "exists", "value": True}]
    filter_uls = [{"attribute": "candid", "operator": "exists", "value": False}]

    pps = []
    uls = []
    for alert in alert_list:
        pps += alert.get_tuples("jd", "fid", filters=filter_pps)
        uls += alert.get_values("jd", filters=filter_uls)
        datapoints = ZiDataPointShaperBase().process(
            alert.datapoints, stock=alert.stock
        )
        compiler.add(datapoints, channel="channychan", ttl=None, trace_id=None)

    assert len(set(uls)) < len(uls), "Some upper limits duplicated in alerts"
    assert len(set(pps)) < len(pps), "Some photopoints duplicated in alerts"

    compiler.commit(ingester, 0)
    ingester.updates_buffer.push_updates()

    t0 = dev_context.db.get_collection("t0")
    assert t0.count_documents({"id": {"$gt": 0}}) == len(set(pps))
    assert t0.count_documents({"id": {"$lt": 0}}) == len(set(uls))


@pytest.fixture
def ingestion_handler_with_mongomuxer(mock_context):
    mock_context.add_channel("EXAMPLE_TNS_MSIP", ["ZTF", "ZTF_PUB", "ZTF_PRIV"])
    directive = {
        "channel": "EXAMPLE_TNS_MSIP",
        "ingest": {
            "mux": {
                "unit": "ZiMongoMuxer",
                "combine": [
                    {
                        "unit": "ZiT1Combiner",
                    }
                ],
            },
        },
    }

    with get_handler(mock_context, [IngestDirective(**directive)]) as handler:
        yield handler


def _ingest(handler: ChainedIngestionHandler, alert: AmpelAlertProtocol):
    with handler.ingester.group():
        handler.ingest(alert.datapoints, filter_results=[(0, True)], stock_id=alert.stock)
    handler.ingester.flush()


def test_out_of_order_ingestion(
    mock_context, ingestion_handler_with_mongomuxer, alerts
):
    """
    Returned alert content does not depend on whether photopoints
    were already committed to the database
    """

    alert_list = list(alerts())
    assert alert_list[-1].datapoints[0]["jd"] > alert_list[-2].datapoints[0]["jd"]

    in_order = {
        idx: _ingest(ingestion_handler_with_mongomuxer, alert_list[idx])
        for idx in (-3, -1, -2)
    }

    # clean up mutations
    mock_context.db.get_collection("t0").delete_many({})
    mock_context.db.get_collection("t1").delete_many({})
    alert_list = list(alerts())

    out_of_order = {
        idx: _ingest(ingestion_handler_with_mongomuxer, alert_list[idx])
        for idx in (-3, -2, -1)
    }

    for idx in sorted(in_order.keys()):
        assert in_order[idx] == out_of_order[idx]

