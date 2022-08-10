#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/t2/T2LightCurveSummary.py
# License:             BSD-3-Clause
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                16.12.2020
# Last Modified Date:  16.12.2020
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>

from typing import Any
from ampel.types import UBson
from ampel.abstract.AbsLightCurveT2Unit import AbsLightCurveT2Unit
from ampel.struct.UnitResult import UnitResult
from ampel.view.LightCurve import LightCurve


class T2LightCurveSummary(AbsLightCurveT2Unit):
    """
    Calculate summary quantities from the light curve.

    This can be signficantly more efficient than calculating the same
    quantities at T3 level for channels that select only a subset of
    datapoints for each stock.
    """

    #: Fields to extract from the latest candidate
    cols: list[str] = [
        "drb",
        "ra",
        "dec",
        "magpsf",
        "sgscore1",
        "distnr",
        "distpsnr1",
    ]
    #: Minimum magnitude of nondetections
    limiting_magnitude: float = 19.5

    def process(self, lightcurve: LightCurve) -> UBson | UnitResult:
        result: dict[str, Any] = {
            "num_detections": len(lightcurve.get_photopoints() or []),
        }
        if (pps := lightcurve.get_photopoints()):
            first, latest = pps[0]["body"], pps[-1]["body"]
            result["first_detection"] = first["jd"]
            result["ra_dis"], result["dec_dis"] = first["ra"], first["dec"]

            result["last_detection"] = latest["jd"]
            for k in self.cols:
                result[k] = latest.get(k)

            # find the last strong upper limit before the first detection
            if last_significant_nondetection := next(
                reversed(
                    lightcurve.get_upperlimits(
                        [
                            {
                                "attribute": "jd",
                                "operator": "<",
                                "value": result["first_detection"],
                            },
                            {
                                "attribute": "diffmaglim",
                                "operator": ">",
                                "value": self.limiting_magnitude,
                            },
                        ]
                    )
                    or []
                ),
                None,
            ):
                for field in ("jd", "fid", "diffmaglim"):
                    result[
                        f"last_significant_nondetection_{field}"
                    ] = last_significant_nondetection["body"][field]

        return result
