import pytest

from ampel.model.UnitModel import UnitModel
from ampel.dev.DevAmpelContext import DevAmpelContext
from ampel.ztf.t1.ZiT1RetroCombiner import ZiT1RetroCombiner
from ampel.ztf.t1.ZiT1Combiner import ZiT1Combiner
from ampel.ztf.ingest.ZiDataPointShaper import ZiDataPointShaperBase


@pytest.fixture
def alert_datapoints(upper_limit_alerts):
    shaper = ZiDataPointShaperBase()
    return [
        shaper.process(alert.datapoints, alert.stock) for alert in upper_limit_alerts
    ]


def test_retro_combiner(mock_context: DevAmpelContext, alert_datapoints, ampel_logger):
    mock_context.register_unit(ZiT1RetroCombiner)
    combiner = mock_context.loader.new_logical_unit(
        UnitModel(unit="ZiT1RetroCombiner", config={"access": [], "policy": []}),
        logger=ampel_logger,
        sub_type=ZiT1RetroCombiner,
    )
    for datapoints in alert_datapoints:
        results = combiner.combine(datapoints)
        assert len(results) == len([dp for dp in datapoints if "ZTF_DP" in dp["tag"]])
        get_dp = lambda id: next(dp for dp in datapoints if dp["id"] == id)
        for result in combiner.combine(datapoints):
            labels = [
                next(dp for dp in datapoints if dp["id"] == i)["tag"][-1]
                for i in result.dps
            ]
            assert labels[-1] == "ZTF_DP"
            jds = [get_dp(i)["body"]["jd"] for i in result.dps]
            assert jds == sorted(jds)


def test_simple_combiner(mock_context: DevAmpelContext, alert_datapoints, ampel_logger):
    mock_context.register_unit(ZiT1Combiner)
    combiner = mock_context.loader.new_logical_unit(
        UnitModel(unit="ZiT1Combiner", config={"access": [], "policy": []}),
        logger=ampel_logger,
        sub_type=ZiT1Combiner,
    )
    for datapoints in alert_datapoints:
        result = combiner.combine(datapoints)
        get_dp = lambda id: next(dp for dp in datapoints if dp["id"] == id)
        jds = [get_dp(i)["body"]["jd"] for i in result.dps]
        assert jds == sorted(jds)
