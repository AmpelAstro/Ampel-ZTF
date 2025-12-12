from pathlib import Path

import yaml

from ampel.core.AmpelContext import AmpelContext
from ampel.log.AmpelLogger import AmpelLogger
from ampel.model.UnitModel import UnitModel
from ampel.ztf.t0.DecentFilter import DecentFilter


def test_decentfilter_star_in_gaia(
    mock_context: AmpelContext,
    ampel_logger: AmpelLogger,
):
    with open(Path(__file__).parent / "test-data" / "decentfilter_config.yaml") as f:
        config = yaml.safe_load(f)
    unit: DecentFilter = mock_context.loader.new_logical_unit(
        UnitModel(unit="DecentFilter", config=config),
        logger=ampel_logger,
        sub_type=DecentFilter,
    )
    assert unit.is_star_in_gaia(
        {"ra": 0.009437700971970959, "dec": -0.0008937364197194631}
    )
    assert not unit.is_star_in_gaia({"ra": 0, "dec": 0})
