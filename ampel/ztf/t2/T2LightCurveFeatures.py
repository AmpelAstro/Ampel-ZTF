#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/t2/T2LightCurveFeatures.py
# License:             BSD-3-Clause
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                15.04.2021
# Last Modified Date:  15.04.2021
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>

from collections.abc import Iterable
from contextlib import suppress
from typing import Any

import light_curve
import numpy as np

from ampel.abstract.AbsStateT2Unit import AbsStateT2Unit
from ampel.abstract.AbsTabulatedT2Unit import AbsTabulatedT2Unit
from ampel.content.DataPoint import DataPoint
from ampel.content.T1Document import T1Document


class T2LightCurveFeatures(AbsStateT2Unit, AbsTabulatedT2Unit):
    """
    Calculate various features of the light curve using the light-curve
    package described in https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M%2F/abstract
    """

    #: Features to extract from the light curve.
    #: See: https://docs.rs/light-curve-feature/0.2.2/light_curve_feature/features/index.html
    features: dict[str, None | dict[str, Any]] = {
        "InterPercentileRange": {"quantile": 0.25},
        "LinearFit": None,
        "StetsonK": None,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extractor = light_curve.Extractor(
            *(getattr(light_curve, k)(**(v or {})) for k, v in self.features.items())
        )

    def process(
        self, compound: T1Document, datapoints: Iterable[DataPoint]
    ) -> dict[str, float]:
        table = self.get_flux_table(datapoints).group_by("band")
        # lightcurve package expects magnitudes
        table["mag"] = -2.5 * np.log10(table["flux"]) + table["zp"]
        table["magerr"] = np.abs(table["fluxerr"] / table["flux"]) * (2.5 / np.log(10))

        result = {}
        for band, bandtable in zip(
            table.groups.keys["band"], table.groups, strict=True
        ):
            with suppress(ValueError):  # raised if too few points
                result.update(
                    {
                        f"{k}_{band}": v
                        for k, v in zip(
                            self.extractor.names,
                            self.extractor(
                                bandtable["time"], bandtable["mag"], bandtable["magerr"]
                            ),
                            strict=False,
                        )
                    }
                )
        return result
