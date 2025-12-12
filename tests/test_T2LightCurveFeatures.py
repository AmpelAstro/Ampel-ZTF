from pathlib import Path

import pytest

from ampel.ztf.dev.ZTFAlert import ZTFAlert

T2LightCurveFeatures = pytest.importorskip("ampel.ztf.t2.T2LightCurveFeatures")


@pytest.fixture
def lightcurve(mock_context):
    path = str(Path(__file__).parent / "test-data" / "ZTF20abyfpze.avro")

    return ZTFAlert.to_transientview(file_path=path)


def test_features(lightcurve, ampel_logger, snapshot):
    t2 = T2LightCurveFeatures.T2LightCurveFeatures(
        logger=ampel_logger, tabulator=[{"unit": "ZTFT2Tabulator"}]
    )
    result = t2.process(None, lightcurve.get_photopoints())
    assert result == snapshot
    for prefix in t2.extractor.names:
        assert any(k.startswith(prefix) for k in result)
