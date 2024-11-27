import logging
from pathlib import Path

import pytest

from ampel.ztf.dev.ZTFAlert import ZTFAlert

T2LightCurveFeatures = pytest.importorskip("ampel.ztf.t2.T2LightCurveFeatures")


@pytest.fixture
def lightcurve(mock_context):
    path = str(Path(__file__).parent / "test-data" / "ZTF20abyfpze.avro")

    return ZTFAlert.to_lightcurve(file_path=path)


def test_features(lightcurve):
    t2 = T2LightCurveFeatures.T2LightCurveFeatures(logger=logging.getLogger())
    result = t2.process(lightcurve)
    for prefix in t2.extractor.names:
        assert any(k.startswith(prefix) for k in result)
