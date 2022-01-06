import itertools, os, fastavro, pytest, before_after
from ampel.dev.DevAmpelContext import DevAmpelContext
from ampel.content.DataPoint import DataPoint
from ampel.types import DataPointId
from collections import defaultdict
from pymongo.operations import UpdateOne

from ampel.core.AmpelContext import AmpelContext
from ampel.dev.UnitTestAlertSupplier import UnitTestAlertSupplier
from ampel.ingest.ChainedIngestionHandler import ChainedIngestionHandler
from ampel.ingest.T0Compiler import T0Compiler
from ampel.log.AmpelLogger import DEBUG, AmpelLogger
from ampel.model.ingest.IngestDirective import IngestDirective
from ampel.model.UnitModel import UnitModel
from ampel.mongo.update.DBUpdatesBuffer import DBUpdatesBuffer
from ampel.mongo.update.MongoT0Ingester import MongoT0Ingester
from ampel.secret.AmpelVault import AmpelVault
from ampel.secret.DictSecretProvider import DictSecretProvider
from ampel.ztf.alert.ZiAlertSupplier import ZiAlertSupplier
from ampel.ztf.ingest.ZiArchiveMuxer import ZiArchiveMuxer
from ampel.ztf.ingest.ZiCompilerOptions import ZiCompilerOptions
from ampel.ztf.ingest.ZiDataPointShaper import ZiDataPointShaperBase
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.ztf.ingest.ZiMongoMuxer import ZiMongoMuxer

def _make_muxer(context: AmpelContext, model: UnitModel) -> ZiArchiveMuxer:
    run_id = 0
    logger = AmpelLogger.get_logger()
    updates_buffer = DBUpdatesBuffer(context.db, run_id=run_id, logger=logger)

    muxer = context.loader.new_context_unit(
        model=model,
        sub_type=ZiArchiveMuxer,
        context=context,
        logger=logger,
        updates_buffer=updates_buffer,
    )

    return muxer


def get_supplier(loader):
    supplier = ZiAlertSupplier(
        deserialize="avro", loader=UnitTestAlertSupplier(alerts=list(loader))
    )
    return supplier


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


@pytest.fixture()
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
    objectId = list(photopoints.keys())[0]
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
def test_instantiate(patch_mongo, dev_context: AmpelContext, model):
    _make_muxer(dev_context, model)


@pytest.fixture
def mock_get_photopoints(mocker, consolidated_alert):
    # mock get_photopoints to return first alert
    mocker.patch(
        "ampel.ztf.ingest.ZiArchiveMuxer.ZiArchiveMuxer.get_photopoints",
        return_value=consolidated_alert,
    )


@pytest.fixture
def mock_archive_muxer(patch_mongo, dev_context, mock_get_photopoints):
    ingester = _make_muxer(
        dev_context, UnitModel(unit="ZiArchiveMuxer", config={"history_days": 30})
    )
    return ingester


@pytest.fixture
def t0_ingester(patch_mongo, dev_context: DevAmpelContext):
    run_id = 0
    logger = AmpelLogger.get_logger()
    updates_buffer = DBUpdatesBuffer(dev_context.db, run_id=run_id, logger=logger)
    ingester = MongoT0Ingester(updates_buffer=updates_buffer)
    compiler = T0Compiler(tier=0, run_id=run_id)
    muxer = ZiMongoMuxer(context=dev_context, updates_buffer=updates_buffer, logger=logger)
    return ingester, compiler, muxer


def test_get_earliest_jd(
    t0_ingester: tuple[MongoT0Ingester, T0Compiler, ZiMongoMuxer], mock_archive_muxer, alerts
):
    """earliest jd is stable under out-of-order ingestion"""

    alert_list = list(alerts())

    ingester, compiler, muxer = t0_ingester

    for i in [2, 0, 1]:

        datapoints = ZiDataPointShaperBase().process(
            alert_list[i].datapoints, stock=alert_list[i].stock
        )
        compiler.add(datapoints, channel="EXAMPLE_TNS_MSIP", trace_id=0)
        compiler.commit(ingester, 0)

        assert mock_archive_muxer.get_earliest_jd(
            alert_list[i].stock, datapoints
        ) == min(
            dp["body"]["jd"] for dp in [el for el in datapoints if el["body"].get("candid")]
        ), "min jd is min jd of last ingested alert"


def get_handler(context, directives, run_id=0) -> ChainedIngestionHandler:
    logger = AmpelLogger.get_logger(console={"level": DEBUG})
    updates_buffer = DBUpdatesBuffer(context.db, run_id=run_id, logger=logger)
    return ChainedIngestionHandler(
        context=context,
        logger=logger,
        run_id=0,
        updates_buffer=updates_buffer,
        directives=directives,
        # do not attempt to sort datapoint ids
        compiler_opts=ZiCompilerOptions(t1={'tag': 'ZTF', 'sort': False}),
        shaper=UnitModel(unit="ZiDataPointShaper"),
        trace_id={},
        tier=0,
    )


def test_integration(patch_mongo, dev_context, mock_get_photopoints, alerts):
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

    handler = get_handler(dev_context, [IngestDirective(**directive)])

    t0 = dev_context.db.get_collection("t0")
    t1 = dev_context.db.get_collection("t1")
    t2 = dev_context.db.get_collection("t2")
    assert t0.count_documents({}) == 0

    alert_list = list(alerts())

    handler.ingest(
        alert_list[1].datapoints,
        stock_id=alert_list[1].stock,
        filter_results=[(0, True)],
    )
    handler.updates_buffer.push_updates()

    assert ZiArchiveMuxer.get_photopoints.called_once()

    # note lack of handler.updates_buffer.push_updates() here;
    # ZiAlertContentIngester has to be synchronous to deal with superseded
    # photopoints
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
            [DictSecretProvider({"ztf/archive/token": token})]
            + mock_context.loader.vault.providers
        ),
    )
    yield token


def test_get_photopoints_from_api(mock_context, archive_token):
    """
    ZiT1ArchivalCompoundIngester can communicate with the archive service
    """
    muxer = _make_muxer(
        mock_context, UnitModel(unit="ZiArchiveMuxer", config={"history_days": 30})
    )
    alert = muxer.get_photopoints("ZTF18abcfcoo", before_jd=2458300)
    assert len(alert["prv_candidates"]) == 10


def test_deduplication(
    dev_context, t0_ingester: tuple[MongoT0Ingester, T0Compiler, ZiMongoMuxer], alerts
):
    """
    Database gets only one copy of each datapoint
    """

    alert_list = list(itertools.islice(alerts(), 1, None))

    ingester, compiler, muxer = t0_ingester
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
        to_insert, to_combine = muxer.process(datapoints, alert.stock)
        compiler.add(to_insert, "channychan", 0)
        compiler.commit(ingester, 0)
        ingester.updates_buffer.push_updates(force=True)

    assert len(set(uls)) < len(uls), "Some upper limits duplicated in alerts"
    assert len(set(pps)) < len(pps), "Some photopoints duplicated in alerts"
    

    t0 = dev_context.db.get_collection("t0")
    assert t0.count_documents({"tag": "ZTF_DP"}) == len(set(pps))
    assert t0.count_documents({"tag": "ZTF_UL"}) == len(set(uls))


@pytest.fixture
def ingestion_handler_with_mongomuxer(mock_context):
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

    return get_handler(mock_context, [IngestDirective(**directive)])


def _ingest(handler: ChainedIngestionHandler, alert: AmpelAlertProtocol):
    handler.ingest(alert.datapoints, filter_results=[(0, True)], stock_id=alert.stock)
    assert len(updates := handler.updates_buffer.db_ops["t1"]) == 1
    update = updates[0]
    assert isinstance(update, UpdateOne)
    dps = update._doc["$setOnInsert"]["dps"]
    handler.updates_buffer.push_updates()
    return dps


def test_out_of_order_ingestion(
    mock_context, ingestion_handler_with_mongomuxer, alerts
):
    """
    Returned alert content does not depend on whether photopoints
    were already committed to the database
    """

    alert_list = list(alerts())
    assert alert_list[-1].datapoints[0]["jd"] > alert_list[-2].datapoints[0]["jd"]

    ingestion_handler_with_mongomuxer.logger.info("in order")
    in_order = {
        idx: _ingest(ingestion_handler_with_mongomuxer, alert_list[idx])
        for idx in (-3, -1, -2)
    }
    in_order_dps: dict[DataPointId, DataPoint] = {
        dp["id"]: dp for dp in mock_context.db.get_collection("t0").find()
    }

    # clean up mutations
    mock_context.db.get_collection("t0").delete_many({})
    mock_context.db.get_collection("t1").delete_many({})
    alert_list = list(alerts())

    ingestion_handler_with_mongomuxer.logger.info("out of order")
    out_of_order = {
        idx: _ingest(ingestion_handler_with_mongomuxer, alert_list[idx])
        for idx in (-3, -2, -1)
    }
    out_of_order_dps: dict[DataPointId, DataPoint] = {
        dp["id"]: dp for dp in mock_context.db.get_collection("t0").find()
    }

    def resolve_dps(dps_ids, dps):
        t0 = min(dp["body"]["jd"] for dp in dps.values())
        return [
            ((dp := dps[dp_id])["body"]["jd"], dp["body"].get("candid"))
            for dp_id in dps_ids
        ]

    for idx in sorted(in_order.keys()):
        in_order_jds = [in_order_dps[id]["body"]["jd"] for id in in_order[idx]]
        out_of_order_jds = [
            out_of_order_dps[id]["body"]["jd"] for id in out_of_order[idx]
        ]
        assert in_order_jds == out_of_order_jds
        assert out_of_order_jds == sorted(out_of_order_jds)
        assert resolve_dps(in_order[idx], in_order_dps) == resolve_dps(
            out_of_order[idx], out_of_order_dps
        )


def test_superseded_candidates_serial(
    mock_context: AmpelContext,
    ingestion_handler_with_mongomuxer: ChainedIngestionHandler,
    superseded_alerts,
):
    """
    Photopoints are superseded by points from newer alerts with the same jd,rcid
    """

    alerts = list(reversed(list(superseded_alerts())))

    assert alerts[0].datapoints[0]["jd"] == alerts[1].datapoints[0]["jd"]
    candids = [alert.datapoints[0]["candid"] for alert in alerts]
    assert candids[0] < candids[1]

    dps = [_ingest(ingestion_handler_with_mongomuxer, alert) for alert in alerts]

    pp_db = mock_context.db.get_collection("t0").find_one(
        {"body.candid": candids[0]},
    )

    assert "SUPERSEDED" in pp_db["tag"], f"{candids[0]} marked as superseded in db"


@pytest.mark.parametrize("ordering", list(itertools.permutations(range(3))))
def test_superseded_candidates_concurrent(mock_context, superseded_alerts, ordering):
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

    alerts = list(reversed(list(superseded_alerts())))
    assert alerts[0].datapoints[0]["jd"] == alerts[1].datapoints[0]["jd"]
    candids = [alert.datapoints[0]["candid"] for alert in alerts]

    ingesters = [
        get_handler(mock_context, [IngestDirective(**directive)], i)
        for i in range(len(alerts))
    ]

    assert len(alerts) == 3

    def _ingest(indexes: list[int]):
        for i in indexes:
            next(iter(ingesters[i]._mux_cache.values())).index = i
            ingesters[i].ingest(
                alerts[i].datapoints,
                filter_results=[(0, True)],
                stock_id=alerts[i].stock,
            )
            ingesters[i].updates_buffer.push_updates()

    # simulate real-world race conditions by running an entire ingestion immediately after
    # one ingester retrieves existing datapoints, but before it pushes any updates
    #
    # this creates sequences like the following:
    # 0 begin 1391345455815015017
    # 2 begin 1391345455815015019
    # 1 begin 1391345455815015018
    # 1 end  1391345455815015018
    # 2 end  1391345455815015019
    # 0 end  1391345455815015017
    def ingest(indexes: list[int], interleave=True):
        if interleave and len(indexes) > 1:
            with before_after.after(
                "ampel.ztf.ingest.ZiMongoMuxer.ZiMongoMuxer._get_dps",
                lambda *args: ingest(indexes[1:], interleave),
            ):
                _ingest(indexes[:1])
        else:
            _ingest(indexes)

    ingest(ordering)

    t0 = mock_context.db.get_collection("t0")

    def assert_superseded(old, new):
        doc = t0.find_one({"body.candid": old})
        newdoc = t0.find_one({"body.candid": new})
        meta = doc.get("meta", [])
        assert (
            "SUPERSEDED" in doc["tag"]
            and len(
                [
                    m
                    for m in meta
                    if m.get("tag") == "SUPERSEDED"
                    and m.get("extra", {}).get("newId") == newdoc["id"]
                ]
            )
            == 1
        ), f"candid {old} superseded by {new}"

    assert_superseded(candids[0], candids[1])
    assert_superseded(candids[0], candids[2])
    assert_superseded(candids[1], candids[2])
    assert (
        "SUPERSEDED" not in t0.find_one({"body.candid": candids[2]})["tag"]
    ), f"candid {candids[2]} not superseded"
