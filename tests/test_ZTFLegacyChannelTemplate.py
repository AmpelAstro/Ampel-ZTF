from ampel.config.AmpelConfig import AmpelConfig
from ampel.core.UnitLoader import UnitLoader
from ampel.model.ingest.FilterModel import FilterModel
from ampel.model.ingest.MuxModel import MuxModel
from ampel.model.ProcessModel import ProcessModel
import pytest
import yaml
from pathlib import Path

from ampel.template.ZTFLegacyChannelTemplate import ZTFLegacyChannelTemplate
from ampel.model.ingest.IngestDirective import IngestDirective
from ampel.log.AmpelLogger import AmpelLogger

@pytest.fixture
def logger():
    return AmpelLogger.get_logger()

@pytest.fixture
def unit_loader(first_pass_config):
    return UnitLoader(AmpelConfig(first_pass_config, freeze=True), db=None, provenance=False)

def test_alert_only(logger, first_pass_config, unit_loader: UnitLoader):
    template = ZTFLegacyChannelTemplate(
        **{
            "channel": "EXAMPLE_TNS_MSIP",
            "version": 0,
            "contact": "ampel@desy.de",
            "active": True,
            "auto_complete": False,
            "template": "ztf_uw_public",
            "t0_filter": {"unit": "NoFilter"},
        }
    )
    process = template.get_processes(logger=logger, first_pass_config=first_pass_config)[0]
    assert process["tier"] == 0
    with unit_loader.validate_unit_models():
        directive = IngestDirective(**process["processor"]["config"]["directives"][0])
    assert isinstance(directive.filter, FilterModel)
    assert isinstance(directive.ingest.mux, MuxModel)
    assert len(directive.ingest.mux.combine) == 1
    assert len(units := directive.ingest.mux.combine[0].state_t2) == 1
    assert units[0].unit == "T2LightCurveSummary"
    assert directive.ingest.combine is None

    with unit_loader.validate_unit_models():
        ProcessModel(**(process | {"version": 0}))

def test_alert_t2(logger, first_pass_config):
    """
    With live_history disabled, T2s run on alert history only
    """
    template = ZTFLegacyChannelTemplate(
        **{
            "channel": "EXAMPLE_TNS_MSIP",
            "contact": "ampel@desy.de",
            "version": 0,
            "active": True,
            "auto_complete": False,
            "template": "ztf_uw_public",
            "t0_filter": {"unit": "NoFilter"},
            "t2_compute": {"unit": "DemoLightCurveT2Unit",},
            "live_history": False,
        }
    )
    process = template.get_processes(logger=logger, first_pass_config=first_pass_config)[0]
    assert process["tier"] == 0
    directive = IngestDirective(**process["processor"]["config"]["directives"][0])
    assert directive.ingest.mux is None
    assert len(directive.ingest.combine) == 1
    assert len(units := directive.ingest.combine[0].state_t2) == 2
    assert {u.unit for u in units} == {"DemoLightCurveT2Unit", "T2LightCurveSummary"}

def test_archive_t2(logger, first_pass_config):
    """
    With archive_history disabled, T2s run on alert history only
    """
    template = ZTFLegacyChannelTemplate(
        **{
            "channel": "EXAMPLE_TNS_MSIP",
            "contact": "ampel@desy.de",
            "version": 0,
            "active": True,
            "auto_complete": False,
            "template": "ztf_uw_public",
            "t0_filter": {"unit": "NoFilter"},
            "t2_compute": {"unit": "DemoLightCurveT2Unit",},
            "live_history": True,
            "archive_history": 42,
        }
    )
    process = template.get_processes(logger=logger, first_pass_config=first_pass_config)[0]
    assert process["tier"] == 0
    directive = IngestDirective(**process["processor"]["config"]["directives"][0])
    assert isinstance(directive.filter, FilterModel)
    assert isinstance(directive.ingest.mux, MuxModel)
    assert directive.ingest.mux.unit == "ChainedT0Muxer"
    assert len(directive.ingest.mux.config["muxers"]) == 2
    assert len(directive.ingest.mux.combine) == 1
    assert len(units := directive.ingest.mux.combine[0].state_t2) == 2
    assert {u.unit for u in units} == {"DemoLightCurveT2Unit", "T2LightCurveSummary"}
    assert directive.ingest.combine is None
