from functools import partial
from pathlib import Path

import pytest
import yaml

from ampel.alert.load.TarAlertLoader import TarAlertLoader


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
def first_pass_config():
    with open(Path(__file__).parent / "test-data" / "testing-config.yaml") as f:
        return yaml.safe_load(f)
