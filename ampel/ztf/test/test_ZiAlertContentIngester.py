import asyncio
import copy
import itertools
import os
import pickle
import socket
from functools import partial
from pathlib import Path

import mongomock
import pymongo
import pytest

from ampel.alert.load.TarAlertLoader import TarAlertLoader
from ampel.config.AmpelConfig import AmpelConfig
from ampel.db.DBUpdatesBuffer import DBUpdatesBuffer
from ampel.dev.DevAmpelContext import DevAmpelContext
from ampel.log.AmpelLogger import AmpelLogger
from ampel.log.LogsBufferDict import LogsBufferDict
from ampel.util import concurrent
from ampel.ztf.alert.ZiAlertSupplier import ZiAlertSupplier
from ampel.ztf.ingest.ZiAlertContentIngester import ZiAlertContentIngester


@pytest.fixture
def patch_mongo(monkeypatch):
    monkeypatch.setattr("ampel.db.AmpelDB.MongoClient", mongomock.MongoClient)


@pytest.fixture
def dev_context():
    config = AmpelConfig.load(
        Path(__file__).parent / "test-data" / "testing-config.yaml",
    )
    custom_conf = {}
    if "MONGO_HOSTNAME" in os.environ:
        custom_conf[
            "resource.mongo"
        ] = f"mongodb://{os.environ['MONGO_HOSTNAME']}:{os.environ.get('MONGO_PORT', 27017)}"
    try:
        return DevAmpelContext.new(
            config=config, purge_db=True, custom_conf=custom_conf
        )
    except pymongo.errors.ServerSelectionTimeoutError:
        raise pytest.skip(f"No mongod listening on {(custom_conf or config).get('resource.mongo')}")


@pytest.fixture
def avro_packets():
    """
    4 alerts for a random AGN, widely spaced:
    
    ------------------ -------------------------- ------------------------
    candid             detection                  history
    ------------------ -------------------------- ------------------------
    673285273115015035 2018-11-05 06:50:48.001935 29 days, 22:11:31.004165 
    879461413115015009 2019-05-30 11:04:25.996800 0:00:00 
    882463993115015007 2019-06-02 11:08:09.003839 3 days, 0:03:43.007039 
    885458643115015010 2019-06-05 11:00:26.997131 5 days, 23:56:01.000331 
    ------------------ -------------------------- ------------------------
    """
    return partial(
        TarAlertLoader, Path(__file__).parent / "test-data" / "ZTF18abxhyqv.tar.gz"
    )


@pytest.fixture
def superseded_packets():
    """
    Three alerts, received within 100 ms, with the same points but different candids
    """
    return partial(
        TarAlertLoader, Path(__file__).parent / "test-data" / "ZTF18acruwxq.tar.gz"
    )


def _make_ingester(context):
    run_id = 0
    logger = AmpelLogger.get_logger()
    updates_buffer = DBUpdatesBuffer(context.db, run_id=run_id, logger=logger)
    logd = LogsBufferDict({"logs": [], "extra": {}, "err": False,})

    ingester = ZiAlertContentIngester(
        updates_buffer=updates_buffer, logd=logd, run_id=run_id, context=context,
    )

    return ingester


@pytest.fixture
def ingester(patch_mongo, dev_context):
    """
    Set up ZiAlertContentIngester
    """
    return _make_ingester(dev_context)


def get_supplier(loader):
    supplier = ZiAlertSupplier(deserialize="avro")
    supplier.set_alert_source(loader)
    return supplier


def test_deduplication(ingester, avro_packets):
    """
    Database gets only one copy of each datapoint
    """

    alerts = list(get_supplier(itertools.islice(avro_packets(), 1, None)))

    pps = []
    uls = []
    for alert in alerts:
        pps += alert.get_tuples("jd", "fid", data="pps")
        uls += alert.get_values("jd", data="uls")
        ingester.ingest(alert)

    assert len(set(uls)) < len(uls), "Some upper limits duplicated in alerts"
    assert len(set(pps)) < len(pps), "Some photopoints duplicated in alerts"

    ingester.updates_buffer.push_updates()

    t0 = ingester.context.db.get_collection("t0")
    assert t0.count_documents({"_id": {"$gt": 0}}) == len(set(pps))
    assert t0.count_documents({"_id": {"$lt": 0}}) == len(set(uls))


def test_out_of_order_ingestion(ingester, avro_packets):
    """
    Returned alert content does not depend on whether photopoints
    were already committed to the database
    """

    alerts = list(get_supplier(avro_packets()))

    assert alerts[-1].pps[0]["jd"] > alerts[-2].pps[0]["jd"]

    def ingest(alert):
        dps = ingester.ingest(alert)
        ingester.updates_buffer.push_updates()
        return dps

    in_order = {idx: ingest(alerts[idx]) for idx in (-3, -1, -2)}

    # clean up mutations
    ingester.context.db.get_collection("t0").delete_many({})
    alerts = list(get_supplier(avro_packets()))

    out_of_order = {idx: ingest(alerts[idx]) for idx in (-3, -2, -1)}

    for idx in sorted(in_order.keys()):
        assert in_order[idx] == out_of_order[idx]


def test_superseded_candidates_serial(ingester, superseded_packets):
    """
    Photopoints are superseded by points from newer alerts with the same jd,rcid
    """

    alerts = list(reversed(list(get_supplier(superseded_packets()))))

    assert alerts[0].pps[0]["jd"] == alerts[1].pps[0]["jd"]
    candids = [alert.pps[0]["candid"] for alert in alerts]
    assert candids[0] < candids[1]

    dps = [ingester.ingest(alert) for alert in alerts]

    pp_db = ingester.context.db.get_collection("t0").find_one(
        {"_id": candids[0]}, ingester.projection
    )

    assert "SUPERSEDED" in pp_db["tag"], f"{candids[0]} marked as superseded in db"
    assert (
        dps[0][0]["tag"] + ["SUPERSEDED"] == pp_db["tag"]
    ), "data points match database content"


@concurrent.process
def run_ingester(config, port):
    """
    Run ingester in a subprocess.
    """

    ctx = DevAmpelContext.new(config=AmpelConfig(config, freeze=True))
    ingester = _make_ingester(ctx)

    conn = socket.create_connection(("127.0.0.1", port))

    conn.send(b"hola")
    reply = b""
    while chunk := conn.recv(4096):
        reply += chunk

    alert = pickle.loads(reply)
    candid = alert.pps[0]["candid"]
    print(f"pid {os.getpid()} got alert {candid}")
    dps = ingester.ingest(alert)
    ingester.updates_buffer.push_updates()

    return candid, dps


@pytest.mark.asyncio
async def test_superseded_candidates_concurrent(
    dev_context, superseded_packets, unused_tcp_port
):
    """
    Photopoints are marked superseded when alerts are ingested simultaneously
    """

    class Distributor:
        """
        Wait for all clients to connect, then deliver messages all at once
        """

        def __init__(self, payloads):
            self.cond = asyncio.Condition()
            self.payloads = payloads

        async def __call__(self, reader, writer):
            data = await reader.read(100)

            payload = self.payloads.pop()

            # block until all payloads are ready to send
            async with self.cond:
                self.cond.notify_all()
            while self.payloads:
                async with self.cond:
                    await self.cond.wait()

            writer.write(payload)
            await writer.drain()

            writer.close()

    alerts = list(reversed(list(get_supplier(superseded_packets()))))
    assert alerts[0].pps[0]["jd"] == alerts[1].pps[0]["jd"]
    candids = [alert.pps[0]["candid"] for alert in alerts]
    assert candids[0] < candids[1]

    assert dev_context.db.get_collection("t0").find_one({}) is None

    messages = [pickle.dumps(alert) for alert in alerts]

    server = await asyncio.start_server(
        Distributor(messages), "127.0.0.1", unused_tcp_port
    )
    serve = asyncio.create_task(server.start_serving())

    try:
        tasks = [
            run_ingester(dev_context.config.get(), unused_tcp_port)
            for _ in range(len(messages))
        ]
        returns = await asyncio.gather(*tasks)
    finally:
        serve.cancel()

    ingester = _make_ingester(dev_context)

    without = lambda d, ignored_keys: {
        k: v for k, v in d.items() if not k in ignored_keys
    }

    for candid, dps in returns:
        alert = alerts[candids.index(candid)]

        # ensure that returned datapoints match the shaped alert content, save
        # for tags, which can't be known from a single alert
        assert [without(dp, {"tag"}) for dp in dps if dp["_id"] > 0] == sorted(
            [
                without(ingester.project(dp), {"tag"})
                for dp in ingester.pp_shaper.ampelize(copy.deepcopy(alert.pps))
            ],
            key=lambda pp: pp["body"]["jd"],
            reverse=True,
        ), "photopoints match alert content (except tags)"
        assert [without(dp, {"tag"}) for dp in dps if dp["_id"] < 0] == sorted(
            [
                without(ingester.project(dp), {"tag"})
                for dp in ingester.ul_shaper.ampelize(copy.deepcopy(alert.uls))
            ],
            key=lambda pp: pp["body"]["jd"],
            reverse=True,
        ), "upper limits match alert content (except tags)"

    t0 = dev_context.db.get_collection("t0")

    def assert_superseded(old, new):
        doc = t0.find_one({"_id": old})
        assert (
            "SUPERSEDED" in doc["tag"] and new in doc["newId"]
        ), f"candid {old} superseded by {new}"

    assert_superseded(candids[0], candids[1])
    assert_superseded(candids[1], candids[2])
    assert (
        "SUPERSEDED" not in t0.find_one({"_id": candids[2]})["tag"]
    ), f"candid {candids[2]} not superseded"
