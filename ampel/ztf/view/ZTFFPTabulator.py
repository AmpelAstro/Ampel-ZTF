#!/usr/bin/env python
# File              : Ampel-ZTF/ampel/ztf/view/ZTFFPTabulator.py
# License           : BSD-3-Clause
# Author            : jno
# Date              : 26.05.2021
# Last Modified Date: 05.05.2022
# Last Modified By  : jno

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from astropy.table import Table

from ampel.abstract.AbsT2Tabulator import AbsT2Tabulator
from ampel.content.DataPoint import DataPoint
from ampel.types import StockId
from ampel.util.collections import ampel_iter

# from ampel.ztf.util.ZTFIdMapper import ZTFIdMapper

ZTF_BANDPASSES = {
    1: {"name": "ztfg"},
    2: {"name": "ztfr"},
    3: {"name": "ztfi"},
}


class ZTFFPTabulator(AbsT2Tabulator):
    reject_outlier_sigma: float = (
        10**30
    )  # Immediately reject flux outliers beyond. 0 means none
    flux_max: float = 10**30  # Cut flux above this 0 means none
    inclusion_sigma: float = (
        0  # Only include datapoints with pull detection above this limit
    )

    def filter_detections(self, dps: Iterable[DataPoint]) -> Iterable[DataPoint]:
        return [dp for dp in dps if "ZTF_FP" in dp["tag"] and "flux" in dp["body"]]

    def get_flux_table(
        self,
        dps: Iterable[DataPoint],
    ) -> Table:
        flux, fluxerr, jd, fids = self.get_values(
            self.filter_detections(dps), ["flux", "flux_unc", "jd", "fid"]
        )
        filter_names = [ZTF_BANDPASSES[fid]["name"] for fid in fids]

        # Mask data
        bMask = (
            ((np.abs(flux) / fluxerr) < self.reject_outlier_sigma)
            & ((np.abs(flux) / fluxerr) > self.inclusion_sigma)
            & (np.abs(flux) < self.flux_max)
        )

        return Table(
            {
                "time": jd,
                "flux": flux,
                "fluxerr": fluxerr,
                "band": filter_names,
                "zp": [25] * len(filter_names),
                "zpsys": ["ab"] * len(filter_names),
                # FP also contain the zpdiff field - should this be used here, as value or to scale flux values?
            },
            dtype=("float64", "float64", "float64", "str", "int64", "str"),
        )[bMask]

    def get_positions(
        self, dps: Iterable[DataPoint]
    ) -> Sequence[tuple[float, float, float]]:
        det_dps = self.filter_detections(dps)
        return tuple(
            zip(
                self.get_jd(det_dps),
                *self.get_values(det_dps, ["ra", "dec"]),
                strict=False,
            )
        )

    def get_jd(
        self,
        dps: Iterable[DataPoint],
    ) -> Sequence[Any]:
        return self.get_values(dps, ["jd"])[0]

    def get_stock_id(self, dps: Iterable[DataPoint]) -> set[StockId]:
        return set(
            stockid
            for el in dps
            if "ZTF" in el["tag"]
            for stockid in ampel_iter(el["stock"])
        )

    def get_stock_name(self, dps: Iterable[DataPoint]) -> list[str]:
        """
        Methodology for relating FP request ID to official ZTF names not implemented.
        Could be here?
        Now just converting to strings ("name")
        """
        #        return [ZTFIdMapper.to_ext_id(el) for el in self.get_stock_id(dps)]
        return [str(v) for v in self.get_stock_id(dps)]

    @staticmethod
    def get_values(
        dps: Iterable[DataPoint], params: Sequence[str]
    ) -> tuple[Sequence[Any], ...]:
        if tup := tuple(
            map(
                list,
                zip(
                    *(
                        [el["body"][param] for param in params]
                        for el in dps
                        if "ZTF_FP" in el["tag"]
                    ),
                    strict=False,
                ),
            )
        ):
            return tup

        return tuple([[]] * len(params))
