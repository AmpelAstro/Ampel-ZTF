import pytest
from typing import Callable
from ampel.alert.AlertConsumer import AlertConsumer
from ampel.dev.DevAmpelContext import DevAmpelContext
from ampel.template.ZTFLegacyChannelTemplate import ZTFLegacyChannelTemplate
from ampel.abstract.AbsAlertFilter import AbsAlertFilter
from ampel.alert.AlertConsumer import AlertConsumer
from ampel.model.UnitModel import UnitModel


@pytest.fixture
def testing_context(mock_context):
    return mock_context


@pytest.fixture
def forced_photometry_consumer_factory(
    testing_context: DevAmpelContext,
    forced_photometry_loader_model,
    ampel_logger,
    first_pass_config,
):

    template = ZTFLegacyChannelTemplate(
        **{
            "channel": "EXAMPLE_TNS_MSIP",
            "contact": "ampel@desy.de",
            "version": 0,
            "active": True,
            "auto_complete": False,
            "template": "ztf_uw_public",
            "t0_filter": {"unit": "NoFilter"},
            "t2_compute": {
                "unit": "DemoLightCurveT2Unit",
            },
            "live_history": False,
        }
    )
    process = template.get_processes(
        logger=ampel_logger, first_pass_config=first_pass_config
    )[0]
    process["processor"]["config"]["supplier"]["config"][
        "loader"
    ] = forced_photometry_loader_model.dict()

    class NoFilter(AbsAlertFilter):
        def process(self, alert):
            return True

    testing_context.register_unit(NoFilter)

    def create():
        consumer = testing_context.loader.new_context_unit(
            UnitModel(**process["processor"]),
            context=testing_context,
            process_name="testy",
            sub_type=AlertConsumer,
            raise_exc=True,
        )
        consumer.iter_max = 1
        return consumer

    return create


def test_ingest_forced_photometry(
    testing_context: DevAmpelContext,
    forced_photometry_consumer_factory: Callable[..., AlertConsumer],
):
    """
    With live_history disabled, T2s run on alert history only
    """

    consumer = forced_photometry_consumer_factory()
    consumer.run()

    assert (
        testing_context.db.get_collection("t0").count_documents(
            {"body.candid": {"$exists": 1}, "body.magpsf": {"$gt": 0}}
        )
        == 1
    ), "found subtraction candidate"
    assert (
        testing_context.db.get_collection("t0").count_documents(
            {"body.forcediffimflux": {"$exists": 1}, "body.magpsf": {"$exists": 0}}
        )
        == 23
    ), "found forced photometry"
    assert testing_context.db.get_collection("t0").count_documents({}) == 24
