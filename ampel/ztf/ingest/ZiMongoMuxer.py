#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/ingest/ZiMongoMuxer.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                14.12.2017
# Last Modified Date:  25.05.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from bisect import bisect_right
from typing import Any

from ampel.abstract.AbsT0Muxer import AbsT0Muxer
from ampel.content.DataPoint import DataPoint
from ampel.types import DataPointId, StockId
from ampel.util.mappings import unflatten_dict


class ConcurrentUpdateError(Exception):
    """
    Raised when the t0 collection was updated during ingestion
    """

    ...


class ZiMongoMuxer(AbsT0Muxer):
    """
    This class compares info between alert and DB so that only the needed info is ingested later.

    :param alert_history_length: alerts must not contain all available info for a given transient.
    IPAC generated alerts for ZTF for example currently provide a photometric history of 30 days.
    Although this number is unlikely to change, there is no reason to use a constant in code.
    """

    alert_history_length: int = 30

    # Be idempotent for the sake it (not required for prod)
    idempotent: bool = False

    # True: Alert + DB dps will be combined into state
    # False: Only the alert dps will be combined into state
    db_complete: bool = True

    # Standard projection used when checking DB for existing PPS/ULS
    projection = {
        "_id": 0,
        "id": 1,
        "tag": 1,
        "channel": 1,
        "excl": 1,
        "stock": 1,
        "body.jd": 1,
        "body.programid": 1,
        "body.fid": 1,
        "body.rcid": 1,
        "body.magpsf": 1,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # used to check potentially already inserted pps
        self._photo_col = self.context.db.get_collection("t0")
        self._projection_spec = unflatten_dict(self.projection)

    # NB: this 1-liner is a separate method to provide a patch point for race condition testing
    def _get_dps(self, stock_id: None | StockId) -> list[DataPoint]:
        return list(self._photo_col.find({"stock": stock_id}, self.projection))

    def process(
        self, dps: list[DataPoint], stock_id: None | StockId = None
    ) -> tuple[None | list[DataPoint], None | list[DataPoint]]:
        """
        :param dps: datapoints from alert
        :param stock_id: stock id from alert
        """

        # Part 1: gather info from DB and alert
        #######################################

        # New pps/uls lists for db loaded datapoints
        dps_db = self._get_dps(stock_id)

        # Create set with datapoint ids from alert
        ids_dps_alert = {el["id"] for el in dps}

        # python set of ids of datapoints from DB
        ids_dps_db = {el["id"] for el in dps_db}

        # uniquify photopoints by jd, rcid. For duplicate points,
        # choose the one with the larger id
        # (jd, rcid) -> ids
        unique_dps_ids: dict[tuple[float, int], list[DataPointId]] = {}
        # id -> final datapoint
        unique_dps: dict[DataPointId, DataPoint] = {}

        for dp in dps_db + dps:
            # jd alone is actually enough for matching pps reproc, but an upper limit can
            # be associated with multiple stocks at the same jd. here, match also by rcid
            key = (dp["body"]["jd"], dp["body"]["rcid"])

            # print(dp['id'], key, key in unique_dps)

            if target := unique_dps_ids.get(key):
                # insert id in order
                idx = bisect_right(target, dp["id"])
                if idx == 0 or target[idx - 1] != dp["id"]:
                    target.insert(idx, dp["id"])
            else:
                unique_dps_ids[key] = [dp["id"]]

        # build final set of datapoints, preferring entries loaded from the db
        final_dps_set = {v[-1] for v in unique_dps_ids.values()}
        for dp in dps_db + dps:
            if dp["id"] in final_dps_set and dp["id"] not in unique_dps:
                unique_dps[dp["id"]] = dp

        # Part 2: Update new data points that are already superseded
        ############################################################

        # Difference between candids from the alert and candids present in DB
        ids_dps_to_insert = ids_dps_alert - ids_dps_db

        # The union of the datapoints drawn from the db and
        # from the alert will be part of the t1 document
        if self.db_complete:
            # DB might contain datapoints newer than the newest alert dp
            # https://github.com/AmpelProject/Ampel-ZTF/issues/6
            latest_alert_jd = dps[0]["body"]["jd"]
            dps_combine = [
                dp for dp in unique_dps.values() if dp["body"]["jd"] <= latest_alert_jd
            ]

            # Project datapoint the same way whether they were drawn from the db or from the alert.
            if self.idempotent and self.projection:
                for i, el in enumerate(dps_combine):
                    if el in ids_dps_to_insert:
                        dps_combine[i] = self._project(el, self._projection_spec)
        else:
            dps_combine = dps

        return [dp for dp in dps if dp["id"] in ids_dps_to_insert], dps_combine

    def _project(self, doc, projection) -> DataPoint:
        out: dict[str, Any] = {}
        for key, spec in projection.items():
            if key not in doc:
                continue

            if isinstance(spec, dict):
                item = doc[key]
                if isinstance(item, list):
                    out[key] = [self._project(v, spec) for v in item]
                elif isinstance(item, dict):
                    out[key] = self._project(item, spec)
            else:
                out[key] = doc[key]

        return out  # type: ignore[return-value]
