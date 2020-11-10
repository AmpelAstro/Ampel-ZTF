#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/t3/complement/ZTFCutoutImages.py
# Author            : Jakob van Santen <jakob.van.santen@desy.de>
# Date              : 18.09.2020
# Last Modified Date: 18.09.2020
# Last Modified By  : Jakob van Santen <jakob.van.santen@desy.de>

from typing import Iterable, Literal

from ampel.base.AmpelBaseModel import AmpelBaseModel
from ampel.core.AmpelBuffer import AmpelBuffer
from ampel.core.AmpelContext import AmpelContext
from ampel.model.Secret import Secret
from ampel.t3.complement.AbsT3DataAppender import AbsT3DataAppender
from ampel.ztf.archive.ArchiveDB import ArchiveDB


class ZTFCutoutImages(AbsT3DataAppender):
    """
    Add cutout images from ZTF archive database
    """

    auth: Secret[dict] = {"key": "ztf/archive/reader"}  # type: ignore[assignment]
    #: Which detection to retrieve cutouts for
    eligible: Literal["first", "last", "brightest", "all"] = "last"

    def __init__(self, context: AmpelContext, **kwargs) -> None:

        AmpelBaseModel.__init__(self, **kwargs)

        self.archive = ArchiveDB(
            context.config.get(f"resource.ampel-ztf/archive", str),
            connect_args=self.auth.get(),
        )

    def complement(self, records: Iterable[AmpelBuffer]) -> None:
        for record in records:
            if (photopoints := record.get("t0")) is None:
                raise ValueError(f"{type(self).__name__} requires t0 records")
            pps = sorted(
                [pp for pp in photopoints if pp["_id"] > 0],
                key=lambda pp: pp["body"]["jd"],
            )
            if not pps:
                return

            if self.eligible == "last":
                candids = [pps[-1]["_id"]]
            elif self.eligible == "first":
                candids = [pps[0]["_id"]]
            elif self.eligible == "brightest":
                candids = [min(pps, key=lambda pp: pp["body"]["magpsf"])["_id"]]
            elif self.eligible == "all":
                candids = [pp["_id"] for pp in pps]
            cutouts = {candid: self.archive.get_cutout(candid) for candid in candids}

            if "extra" not in record or record["extra"] is None:
                record["extra"] = {self.__class__.__name__: cutouts}
            else:
                record["extra"][self.__class__.__name__] = cutouts
