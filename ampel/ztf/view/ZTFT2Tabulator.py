#!/usr/bin/env python
# File              : Ampel-LSST/ampel/lsst/view/ZTFT2Tabulator.py
# License           : BSD-3-Clause
# Author            : Marcus Fenner <mf@physik.hu-berlin.de>
# Date              : 26.05.2021
# Last Modified Date: 05.05.2022
# Last Modified By  : Marcus Fenner <mf@physik.hu-berlin.de>

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from astropy.table import Table

from ampel.abstract.AbsT2Tabulator import AbsT2Tabulator
from ampel.content.DataPoint import DataPoint
from ampel.types import DataPointId, StockId
from ampel.util.collections import ampel_iter
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
    reject_outlier_sigma: float = (
        10**30
    )  # Immediately reject flux outliers beyond. 0 means none
    flux_max: float = 10**30  # Cut flux above this 0 means none

    # Check for reprocessed datapoints
    check_reprocessing: bool = True

    def filter_detections(self, dps: Iterable[DataPoint]) -> Iterable[DataPoint]:
        dp_ids = {dp["id"]: dp for dp in dps if "tag" in dp and "ZTF" in dp["tag"]}
        if self.check_reprocessing:
            # uniquify photopoints by jd, rcid. For duplicate points, choose the
            # one with the larger id (jd, rcid) -> ids
            unique_dps_ids: dict[tuple[float, int], list[DataPointId]] = {}
            for dp in dp_ids.values():
                # jd alone is actually enough for matching pps reproc, but an upper limit can
                # be associated with multiple stocks at the same jd. here, match also by rcid
                key = (dp["body"]["jd"], dp["body"]["rcid"])

                if target := unique_dps_ids.get(key):
                    # insert id in order
                    idx = bisect_right(target, dp["id"])
                    if idx == 0 or target[idx - 1] != dp["id"]:
                        target.insert(idx, dp["id"])
                else:
                    unique_dps_ids[key] = [dp["id"]]
            final_dps_set = {v[-1] for v in unique_dps_ids.values()}
        else:
            final_dps_set = set(dp_ids.keys())
        return [dp for dp in dps if dp["id"] in final_dps_set]

    def get_flux_table(
        self,
        dps: Iterable[DataPoint],
    ) -> Table:
        magpsf, sigmapsf, jd, fids = self.get_values(
            self.filter_detections(dps), ["magpsf", "sigmapsf", "jd", "fid"]
        )
        filter_names = [ZTF_BANDPASSES[fid]["name"] for fid in fids]
        # signs = [signdict[el] for el in isdiffpos]
        flux = np.asarray([10 ** (-((mgpsf) - 25) / 2.5) for mgpsf in magpsf])
        fluxerr = np.abs(flux * (-np.asarray(sigmapsf) / 2.5 * np.log(10)))

        # Mask data
        bMask = ((np.abs(flux) / fluxerr) < self.reject_outlier_sigma) & (
            np.abs(flux) < self.flux_max
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
        return [ZTFIdMapper.to_ext_id(el) for el in self.get_stock_id(dps)]

    def get_values(
        self, dps: Iterable[DataPoint], params: Sequence[str]
    ) -> tuple[Sequence[Any], ...]:
        if tup := tuple(
            map(
                list,
                zip(
                    *(
                        [el["body"][param] for param in params]
                        for el in self.filter_detections(dps)
                    ),
                    strict=False,
                ),
            )
        ):
            return tup
        return ([],) * len(params)
