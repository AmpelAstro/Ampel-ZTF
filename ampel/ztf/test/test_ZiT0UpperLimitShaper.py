import pytest

from ampel.ztf.ingest.ZiT0UpperLimitShaper import ZiT0UpperLimitShaper


def test_identity():
    assert (
        ZiT0UpperLimitShaper().identity(
            {
                "diffmaglim": 19.024799346923828,
                "fid": 2,
                "jd": 2458089.7405324,
                "pid": 335240532815,
                "programid": 0,
            }
        )
        == -3352405322819025
    )
