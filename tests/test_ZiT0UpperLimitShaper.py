import pytest
import random

from ampel.ztf.ingest.ZiDataPointShaper import ZiDataPointShaperBase
from ampel.ztf.alert.ZiAlertSupplier import ZiAlertSupplier

# metafixture as suggested in https://github.com/pytest-dev/pytest/issues/349#issuecomment-189370273
@pytest.fixture(params=["upper_limit_alerts", "forced_photometry_alerts"])
def alerts(request):
    yield request.getfixturevalue(request.param)


def test_id_stability(alerts: ZiAlertSupplier, snapshot):
    shaper = ZiDataPointShaperBase()
    alert = next(alerts)
    # NB: process a copy, as ZiDataPointShaper mutates its input
    dps = shaper.process([dict(dp) for dp in alert.datapoints[:5]], alert.stock)
    datapoint_ids = [[dp["tag"][-1], dp["id"]] for dp in dps]
    assert datapoint_ids == snapshot, "datapoint ids are deterministic"

    shuffled_dps = shaper.process(
        [dict(random.sample(list(dp.items()), len(dp))) for dp in alert.datapoints[:5]],
        alert.stock + 1,
    )
    assert [
        [dp["tag"][-1], dp["id"]] for dp in shuffled_dps
    ] == datapoint_ids, "datapoint ids depend only on body content"
