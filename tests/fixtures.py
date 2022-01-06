import json
import subprocess
from functools import partial
from os import environ
from os.path import dirname, join
from pathlib import Path
from time import time
from ampel.log.AmpelLogger import AmpelLogger
from ampel.secret.AmpelVault import AmpelVault

import mongomock
import pymongo
import pytest
import yaml

from ampel.alert.load.TarAlertLoader import TarAlertLoader
from ampel.config.AmpelConfig import AmpelConfig
from ampel.dev.DevAmpelContext import DevAmpelContext
from ampel.secret.PotemkinSecretProvider import PotemkinSecretProvider
from ampel.model.UnitModel import UnitModel
from ampel.ztf.alert.ZiAlertSupplier import ZiAlertSupplier


@pytest.fixture
def patch_mongo(monkeypatch):
    monkeypatch.setattr("ampel.core.AmpelDB.MongoClient", mongomock.MongoClient)


@pytest.fixture(scope="session")
def mongod(pytestconfig):
    if "MONGO_PORT" in environ:
        yield "mongodb://localhost:{}".format(environ["MONGO_PORT"])
        return

    if not pytestconfig.getoption("--integration"):
        raise pytest.skip("integration tests require --integration flag")
    try:
        container = (
            subprocess.check_output(["docker", "run", "--rm", "-d", "-P", "mongo:4.4"])
            .strip()
            .decode()
        )
    except FileNotFoundError:
        pytest.skip("integration tests require docker")
    for _ in range(10):
        try:
            subprocess.check_call(
                [
                    "docker",
                    "exec",
                    container,
                    "sh",
                    "-c",
                    "echo 'db.runCommand({serverStatus:1}).ok' | mongo admin --quiet | grep 1",
                ]
            )
            break
        except subprocess.SubprocessError:
            time.sleep(1)
    else:
        raise subprocess.SubprocessError("mongo failed to start")
    try:
        port = json.loads(subprocess.check_output(["docker", "inspect", container]))[0][
            "NetworkSettings"
        ]["Ports"]["27017/tcp"][0]["HostPort"]
        yield f"mongodb://localhost:{port}"
    finally:
        subprocess.check_call(["docker", "stop", container])


@pytest.fixture
def dev_context(mongod):
    return DevAmpelContext.load(
        config=Path(__file__).parent / "test-data" / "testing-config.yaml",
        purge_db=True,
        custom_conf={"resource.mongo": mongod},
        vault=AmpelVault([PotemkinSecretProvider()]),
    )


@pytest.fixture
def mock_context(patch_mongo):
    return DevAmpelContext.load(
        config=Path(__file__).parent / "test-data" / "testing-config.yaml",
        purge_db=True,
        vault=AmpelVault([PotemkinSecretProvider()]),
    )


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
        TarAlertLoader,
        file_path=str(Path(__file__).parent / "test-data" / "ZTF18abxhyqv.tar.gz"),
    )


@pytest.fixture
def superseded_packets():
    """
    Three alerts, received within 100 ms, with the same points but different candids
    """
    return partial(
        TarAlertLoader,
        file_path=str(Path(__file__).parent / "test-data" / "ZTF18acruwxq.tar.gz"),
    )


@pytest.fixture
def forced_photometry_loader_model():
    """
    Sample data for schema 3.5

    An email was also sent to ztfgeneral about a sample of 5934 alert packets containing forced photometry being available at: https://caltech.box.com/s/op3sma88an06nhzftxz4sq9u5nn8ve4c
    The alerts are:
    from the 2021-07-20 night
    from the public programid=1
    contain a limit of 25 forced photometry points (which is different from what the system will ultimately deliver)
    Forced photometry replaces the previous candidates information in the prv_candidates block. The attached PDF lists the changes from the current v3.3 to the new v3.5 schema.
    This should give you scope to start revising Fritz filters, etc. for the new alert packet structure coming soon.
    """

    return UnitModel(
        unit="TarAlertLoader",
        config={
            "file_path": str(
                Path(__file__).parent / "test-data" / "alerts_2021720_programid1.tar.gz"
            ),
        },
    )


@pytest.fixture
def forced_photometry_alerts(mock_context, forced_photometry_loader_model):
    supplier = ZiAlertSupplier(
        deserialize="avro", loader=forced_photometry_loader_model
    )
    return supplier


@pytest.fixture
def upper_limit_alerts(mock_context, forced_photometry_loader_model):
    supplier = ZiAlertSupplier(
        deserialize="avro",
        loader=UnitModel(
            unit="TarAlertLoader",
            config={
                "file_path": str(
                    Path(__file__).parent / "test-data" / "ZTF18abxhyqv.tar.gz"
                )
            },
        ),
    )
    return supplier


@pytest.fixture
def ampel_logger():
    return AmpelLogger.get_logger()


@pytest.fixture
def first_pass_config():
    with open(Path(__file__).parent / "test-data" / "testing-config.yaml") as f:
        return yaml.safe_load(f)
