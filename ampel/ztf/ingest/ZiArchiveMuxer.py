from functools import cached_property
from typing import Any
from collections.abc import Sequence

import backoff, requests # type: ignore
from requests_toolbelt.sessions import BaseUrlSession

from ampel.types import StockId
from ampel.abstract.AbsT0Muxer import AbsT0Muxer
from ampel.abstract.AbsT0Unit import AbsT0Unit
from ampel.content.DataPoint import DataPoint
from ampel.secret.NamedSecret import NamedSecret
from ampel.model.UnitModel import UnitModel
from ampel.ztf.alert.ZiAlertSupplier import ZiAlertSupplier
from ampel.ztf.base.ArchiveUnit import ArchiveUnit
from ampel.ztf.util.ZTFIdMapper import to_ztf_id


class ZiArchiveMuxer(AbsT0Muxer, ArchiveUnit):
    """
    Add datapoints from archived ZTF-IPAC alerts.
    """

    #: Number of days of history to add, relative to the earliest point in the
    #: t0 collection
    history_days: float = 0
    #: Number of days of looking ahead based on the JD contained in alert
    #: Only effect if original alert obtained through time-dependent search
    #:Warning: could interfer if further alserts added ex through ZiMongoMuxer
    future_days: float = 0


    shaper: UnitModel | str = "ZiDataPointShaper"
    archive_token: NamedSecret[str] = NamedSecret(label="ztf/archive/token")

    # Standard projection used when checking DB for existing PPS/ULS
    projection: dict[str, int] = {
        "_id": 1,
        "tag": 1,
        "excl": 1,
        "body.jd": 1,
        "body.fid": 1,
        "body.rcid": 1,
        "body.magpsf": 1,
    }

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

        self._shaper = self.context.loader.new_logical_unit(
            model=UnitModel(unit="ZiDataPointShaper"),
            logger=self.logger,
            sub_type=AbsT0Unit,
        )

        self._t0_col = self.context.db.get_collection("t0", "w")

    def get_earliest_jd(
        self, stock_id: StockId, datapoints: Sequence[DataPoint]
    ) -> float:
        """
        return the smaller of:
          - the smallest jd of any photopoint in datapoints
          - the smallest jd of any photopoint in t0 from the same stock
        """
        from_alert = min(
            (
                dp["body"]["jd"]
                for dp in datapoints
                if dp["id"] > 0 and "ZTF" in dp["tag"]
            )
        )
        if (
            from_db := next(
                self._t0_col.aggregate(
                    [
                        {
                            "$match": {
                                "id": {"$gt": 0},
                                "stock": stock_id,
                                "body.jd": {"$lt": from_alert},
                                "tag": "ZTF",
                            }
                        },
                        {"$group": {"_id": None, "jd": {"$min": "$body.jd"}}},
                    ]
                ),
                {"jd": None},
            )["jd"]
        ) is None:
            return from_alert
        else:
            return min((from_alert, from_db))

    def get_latest_jd(
        self, datapoints: Sequence[DataPoint]
    ) -> float:
        """
        return the largest jd of any photopoint in datapoints
        Note: Not checking the DB - thought the window should be rel to alert?
        """
        return max(
            (
                dp["body"]["jd"]
                for dp in datapoints
                if dp["id"] > 0 and "ZTF" in dp["tag"]
            )
        )


    @backoff.on_exception(
        backoff.expo,
        requests.HTTPError,
        giveup=lambda e: not isinstance(e, requests.HTTPError) or e.response.status_code not in {503, 504, 429, 408},
        max_time=600,
    )
    def get_photopoints(self, ztf_name: str, jd_center: float, time_pre: float, time_post: float) -> dict[str, Any]:
        response = self.session.get(
            f"object/{ztf_name}/photopoints",
            params={
                "jd_end": jd_center+time_post,
                "jd_start": jd_center-time_pre,
                },
        )
        response.raise_for_status()
        return response.json()



    def process(
        self, dps: list[DataPoint], stock_id: None | StockId = None
    ) -> tuple[None | list[DataPoint], None | list[DataPoint]]:
        """
        :param dps: datapoints from alert
        :param stock_id: stock id from alert
        Attempt to determine which pps/uls should be inserted into the t0 collection,
        and which one should be marked as superseded.
        """

        if not stock_id:
            # no new points to add; use input points for combination
            return dps, dps

        # Alert jd, assumed to be latest dp
        alert_jd =  max(
            (
                dp["body"]["jd"]
                for dp in dps
                if dp["id"] > 0 and "ZTF" in dp["tag"]
            )
        )

        # Obtain archive alert based on this
        # - _possibly_ inefficient since we also search within alert time range
        archive_alert_dict = self.get_photopoints(
            to_ztf_id(stock_id), alert_jd, self.history_days, self.future_days
             )
        archive_alert = ZiAlertSupplier.shape_alert_dict(archive_alert_dict)
        archive_dps = self._shaper.process(archive_alert.datapoints, stock_id)
        if len(archive_dps)==0:
            # nothing found in archive
            return dps, dps


        # Create combined state of alert and archive
        # Only dps with a new jd are added. In practice this means that
        # if (for some reason) a detection exists in the archive, but
        # an upper limit is provided in the alert, then the upper limit is used.
        jds_alert = [dp['body']['jd'] for dp in dps]
        extended_dps = sorted(
            dps + [dp for dp in archive_dps if not dp['body']['jd'] in jds_alert],
            key=lambda d: d["body"]["jd"])

        # Potentially one could here check the DB content, both to extend
        # the state and avoid duplicate inserts. Would then also need to
        # check for superseeded dps.
        # There is possibly an advantage of having a "pure" archive state.

        return extended_dps, extended_dps
