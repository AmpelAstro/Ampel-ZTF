import json
import subprocess
from functools import partial
from os import environ
from pathlib import Path
from time import time

import mongomock
import pytest
import yaml

from ampel.alert.load.TarAlertLoader import TarAlertLoader
from ampel.dev.DevAmpelContext import DevAmpelContext
from ampel.secret.AmpelVault import AmpelVault
from ampel.secret.PotemkinSecretProvider import PotemkinSecretProvider


@pytest.fixture()
def _patch_mongo(monkeypatch):
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


@pytest.fixture()
def dev_context(mongod):
    return DevAmpelContext.load(
        config=Path(__file__).parent / "test-data" / "testing-config.yaml",
        purge_db=True,
        custom_conf={"resource.mongo": mongod},
        vault=AmpelVault([PotemkinSecretProvider()]),
    )


@pytest.fixture()
def mock_context(_patch_mongo):
    return DevAmpelContext.load(
        config=Path(__file__).parent / "test-data" / "testing-config.yaml",
        purge_db=True,
        vault=AmpelVault([PotemkinSecretProvider()]),
    )


@pytest.fixture()
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


@pytest.fixture()
def superseded_packets():
    """
    Three alerts, received within 100 ms, with the same points but different candids
    """
    return partial(
        TarAlertLoader,
        file_path=str(Path(__file__).parent / "test-data" / "ZTF18acruwxq.tar.gz"),
    )


@pytest.fixture()
def first_pass_config():
    with open(Path(__file__).parent / "test-data" / "testing-config.yaml") as f:
        return yaml.safe_load(f)
