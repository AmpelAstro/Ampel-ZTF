from datetime import datetime
from ampel.model.UnitModel import UnitModel
import pytest
import os
from ampel.secret.NamedSecret import NamedSecret

from ampel.ztf.alert.load.ZTFHealpixAlertLoader import HealpixSource
from ampel.ztf.alert.ZiHealpixAlertSupplier import ZiHealpixAlertSupplier
from ampel.ztf.alert.HealpixPathSupplier import HealpixPathSupplier
from ampel.ztf.util.ZTFIdMapper import to_ampel_id


@pytest.fixture
def healpix_dict():
    return {
        "nside": 128,
        "pixels": [114276],
        "time": datetime.fromisoformat("2020-12-20 12:43:06"),
    }


@pytest.fixture
def healpix_source(healpix_dict):
    return HealpixSource(**healpix_dict)


@pytest.fixture
def healpix_loader(archive_token):
    return {
        "unit": "ZTFHealpixAlertLoader",
        "config": {
            "history_days": 0.1,
            "future_days": 0.1,
            "archive_token": archive_token,
        },
    }


@pytest.fixture
def archive_token():
    if not (token := os.environ.get("ARCHIVE_TOKEN")):
        pytest.skip("archive test requires api token")
    archive_token: NamedSecret[str] = NamedSecret(label="ztf/archive/token")
    archive_token.set(token)
    return archive_token


def test_healpix_alertsupplier(mock_context, healpix_source, healpix_loader):
    supplier = ZiHealpixAlertSupplier(source=healpix_source, loader=healpix_loader)
    assert supplier.loader == UnitModel(**healpix_loader)
    alerts = list(iter(supplier))
    assert len(alerts) == 3
    assert alerts[0].stock == to_ampel_id("ZTF18aarmuwm")


def test_healpix_alertsupplier2(mock_context, healpix_dict, healpix_loader):
    supplier = ZiHealpixAlertSupplier(source=None, loader=healpix_loader)
    assert supplier.loader == UnitModel(**healpix_loader)
    supplier.set_healpix(**healpix_dict)
    alerts = list(iter(supplier))
    assert len(alerts) == 3
    assert alerts[0].stock == to_ampel_id("ZTF18aarmuwm")


def test_healpixpathsupplier(healpix_loader):
    supplier = HealpixPathSupplier(
        map_url="https://emfollow.docs.ligo.org/userguide/_static/bayestar.fits.gz,0",
        scratch_dir="./",
        pvalue_limit=0.001,
        loader=healpix_loader,
    )
    assert supplier.trigger_time == datetime.fromisoformat("2018-11-01 22:22:46.654437")
    assert supplier.alert_loader.source.pixels == [
        25541713,
        25541715,
        25541718,
        25541721,
        25541724,
        25541726,
        25541748,
        25541749,
        25541750,
        25541751,
        25541757,
        25541759,
        25541930,
        25542016,
        25542018,
    ]
    alerts = list(iter(supplier))
    assert len(alerts) == 0
