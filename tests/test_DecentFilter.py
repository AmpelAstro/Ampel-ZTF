import pytest

from ampel.dev.DevAmpelContext import DevAmpelContext

from ampel.model.UnitModel import UnitModel
from itertools import islice
from pathlib import Path
from ampel.ztf.t0.DecentFilter import DecentFilter

import yaml


@pytest.fixture
def decent_filter(mock_context: DevAmpelContext, ampel_logger) -> DecentFilter:
    with open(Path(__file__).parent / "test-data" / "decentfilter_config.yaml") as f:
        config = yaml.safe_load(f)
    # loosen tspan cut
    config["max_tspan"] = 100
    return mock_context.loader.new_logical_unit(
        UnitModel(unit="DecentFilter", config=config),
        logger=ampel_logger,
        sub_type=DecentFilter,
    )


def test_forced_photometry(decent_filter: DecentFilter, forced_photometry_alerts):

    i = 0
    for i, alert in enumerate(islice(forced_photometry_alerts, 0, None)):
        assert decent_filter.process(alert) is None
