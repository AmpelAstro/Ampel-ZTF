import tarfile
from pathlib import Path

import pytest

from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.ztf.alert.PhotoAlertPlotter import PhotoAlertPlotter
from ampel.ztf.dev.DevAlertConsumer import DevAlertConsumer


@pytest.fixture()
def recent_alerts():
    def gen():
        dap = DevAlertConsumer(alert_filter=None)
        dap.tar_file = tarfile.open(
            Path(__file__).parent.parent / "alerts" / "recent_alerts.tar.gz"
        )
        for item in dap.tar_file:
            yield dap._unpack(item)

    return gen


def test_PhotoAlertPlotter(recent_alerts):
    plotter = PhotoAlertPlotter(interactive=False)

    alert: AmpelAlertProtocol = next(recent_alerts())

    plotter.summary_plot(alert)
