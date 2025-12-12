from ampel.core.AmpelContext import AmpelContext
from ampel.log.AmpelLogger import AmpelLogger
from ampel.model.ingest.FilterModel import FilterModel
from ampel.model.ingest.IngestDirective import IngestDirective
from ampel.model.ingest.MuxModel import MuxModel
from ampel.model.ingest.T1Combine import T1Combine
from ampel.model.ProcessModel import ProcessModel
from ampel.template.ZTFLegacyChannelTemplate import ZTFLegacyChannelTemplate


def test_alert_only(mock_context: AmpelContext, ampel_logger: AmpelLogger):
    template = ZTFLegacyChannelTemplate(
        **{
            "channel": "EXAMPLE_TNS_MSIP",
            "version": 0,
            "contact": "ampel@desy.de",
            "active": True,
            "auto_complete": False,
            "template": "ztf_uw_public",
            "t0_filter": {"unit": "BasicMultiFilter", "config": {"filters": []}},
        }
    )
    process = template.get_processes(
        logger=ampel_logger, first_pass_config=mock_context.config.get()
    )[0]
    assert process["tier"] == 0
    with mock_context.loader.validate_unit_models():
        directive = IngestDirective(**process["processor"]["config"]["directives"][0])
    assert isinstance(directive.filter, FilterModel)
    assert isinstance(directive.ingest.mux, MuxModel)
    assert directive.ingest.mux.combine
    assert len(directive.ingest.mux.combine) == 1
    assert isinstance((combine := directive.ingest.mux.combine[0]), T1Combine)
    assert isinstance((units := combine.state_t2), list)
    assert len(units) == 1
    assert units[0].unit == "T2LightCurveSummary"
    assert directive.ingest.combine is None

    with mock_context.loader.validate_unit_models():
        ProcessModel(**(process | {"version": 0}))

    assert process["processor"]["config"]["compiler_opts"], "compiler options set"


def test_alert_t2(ampel_logger, mock_context: AmpelContext):
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
            "t0_filter": {"unit": "BasicMultiFilter", "config": {"filters": []}},
            "t2_compute": {
                "unit": "DemoLightCurveT2Unit",
            },
            "live_history": False,
        }
    )
    process = template.get_processes(
        logger=ampel_logger, first_pass_config=mock_context.config.get()
    )[0]
    assert process["tier"] == 0
    directive = IngestDirective(**process["processor"]["config"]["directives"][0])
    assert directive.ingest.mux is None
    assert len(directive.ingest.combine) == 1
    assert len(units := directive.ingest.combine[0].state_t2) == 2
    assert {u.unit for u in units} == {"DemoLightCurveT2Unit", "T2LightCurveSummary"}


def test_archive_t2(ampel_logger, mock_context: AmpelContext):
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
            "t0_filter": {"unit": "BasicMultiFilter", "config": {"filters": []}},
            "t2_compute": {
                "unit": "DemoLightCurveT2Unit",
            },
            "live_history": True,
            "archive_history": 42,
        }
    )
    process = template.get_processes(
        logger=ampel_logger, first_pass_config=mock_context.config.get()
    )[0]
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
