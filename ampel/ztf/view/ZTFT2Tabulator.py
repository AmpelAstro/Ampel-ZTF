#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-LSST/ampel/lsst/view/ZTFT2Tabulator.py
# License           : BSD-3-Clause
# Author            : Marcus Fenner <mf@physik.hu-berlin.de>
# Date              : 26.05.2021
# Last Modified Date: 05.05.2022
# Last Modified By  : Marcus Fenner <mf@physik.hu-berlin.de>

from typing import Any, Sequence, Iterable
from ampel.types import StockId

import numpy as np
from ampel.content.DataPoint import DataPoint
from astropy.table import Table

from ampel.abstract.AbsT2Tabulator import AbsT2Tabulator
from ampel.ztf.util.ZTFIdMapper import ZTFIdMapper

ZTF_BANDPASSES = {
    1: {"name": "ztfg"},
    2: {"name": "ztfr"},
    3: {"name": "ztfi"},
}

signdict = {
    "0": -1,
    "f": -1,
    "1": 1,
    "t": 1,
}


class ZTFT2Tabulator(AbsT2Tabulator):
    reject_outlier_sigma: float = 10**30   # Immediately reject flux outliers beyond. 0 means none
    flux_max: float = 10**30         # Cut flux above this 0 means none

    def filter_detections(
        self, dps: Iterable[DataPoint]
    ) -> Iterable[DataPoint]:
        return [dp for dp in dps
                     if 'ZTF' in dp['tag'] and 'magpsf' in dp['body'].keys() ]

    def get_flux_table(
        self,
        dps: Iterable[DataPoint],
    ) -> Table:
        magpsf, sigmapsf, jd, fids = self.get_values(
            self.filter_detections(dps),
            ["magpsf", "sigmapsf", "jd", "fid"]
        )
        filter_names = [ZTF_BANDPASSES[fid]["name"] for fid in fids]
        #signs = [signdict[el] for el in isdiffpos]
        flux = np.asarray(
            [10 ** (-((mgpsf) - 25) / 2.5) for mgpsf in magpsf]
        )
        fluxerr = np.abs(flux * (-np.asarray(sigmapsf) / 2.5 * np.log(10)))

        # Mask data
        bMask = (
                    ((np.abs(flux)/fluxerr) < self.reject_outlier_sigma) &
                    ( np.abs(flux) < self.flux_max)
                )

        return Table(
            {
                "time": jd,
                "flux": flux,
                "fluxerr": fluxerr,
                "band": filter_names,
                "zp": [25] * len(filter_names),
                "zpsys": ["ab"] * len(filter_names),
            },
            dtype=("float64", "float64", "float64", "str", "int64", "str"),
        )[bMask]

    def get_positions(
        self, dps: Iterable[DataPoint]
    ) -> Sequence[tuple[float, float, float]]:
        det_dps = self.filter_detections(dps)
        return tuple(
            zip(self.get_jd(det_dps), *self.get_values(det_dps, ["ra", "dec"]))
        )

    def get_jd(
        self,
        dps: Iterable[DataPoint],
    ) -> Sequence[Any]:
        return self.get_values(dps, ["jd"])[0]

    def get_stock_id(self, dps: Iterable[DataPoint]) -> set[StockId]:
        return set(
            sum(
                [
                    list(stockid)
                    if isinstance(stockid := el["stock"], Sequence) and not isinstance(stockid, (str,bytes))
                    else [stockid]
                    for el in dps
                    if "ZTF" in el["tag"]
                ],
                [],
            )
        )

    def get_stock_name(self, dps: Iterable[DataPoint]) -> list[str]:
        return [ZTFIdMapper.to_ext_id(el) for el in self.get_stock_id(dps)]

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
                        if "ZTF" in el["tag"]
                    )
                ),
            )
        ):
            return tup
        else:
            return tuple([[]] * len(params))
