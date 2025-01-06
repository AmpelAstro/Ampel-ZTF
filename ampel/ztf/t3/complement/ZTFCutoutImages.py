#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/t3/complement/ZTFCutoutImages.py
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                18.09.2020
# Last Modified Date:  18.09.2020
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>

from base64 import b64decode
from collections.abc import Iterable
from typing import Literal

import backoff
import requests
from requests_toolbelt.sessions import BaseUrlSession

from ampel.abstract.AbsBufferComplement import AbsBufferComplement
from ampel.struct.AmpelBuffer import AmpelBuffer
from ampel.struct.T3Store import T3Store


class ZTFCutoutImages(AbsBufferComplement):
    """
    Add cutout images from ZTF archive database
    """

    #: Which detection to retrieve cutouts for
    eligible: Literal["first", "last", "brightest", "all"] = "last"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.session = BaseUrlSession(
            base_url=self.context.config.get(
                "resource.ampel-ztf/archive", str, raise_exc=True
            )
        )

    @backoff.on_exception(
        backoff.expo,
        requests.ConnectionError,
        max_tries=5,
        factor=10,
    )
    @backoff.on_exception(
        backoff.expo,
        requests.HTTPError,
        giveup=lambda e: not isinstance(e, requests.HTTPError)
        or e.response is None
        or e.response.status_code not in {502, 503, 504, 429, 408},
        max_time=60,
    )
    def get_cutout(self, candid: int) -> None | dict[str, bytes]:
        response = self.session.get(f"alert/{candid}/cutouts")
        if response.status_code == 404:
            return None

        response.raise_for_status()
        return {
            k: b64decode(response.json()[k]["stampData"])
            for k in ["cutoutScience", "cutoutTemplate", "cutoutDifference"]
        }

    def complement(self, records: Iterable[AmpelBuffer], t3s: T3Store) -> None:
        for record in records:
            if (photopoints := record.get("t0")) is None:
                raise ValueError(f"{type(self).__name__} requires t0 records")
            pps = sorted(
                [pp for pp in photopoints if pp["id"] > 0],
                key=lambda pp: pp["body"]["jd"],
            )
            if not pps:
                return

            if self.eligible == "last":
                candids = [pps[-1]["id"]]
            elif self.eligible == "first":
                candids = [pps[0]["id"]]
            elif self.eligible == "brightest":
                candids = [min(pps, key=lambda pp: pp["body"]["magpsf"])["id"]]
            elif self.eligible == "all":
                candids = [pp["id"] for pp in pps]
            cutouts = {candid: self.get_cutout(candid) for candid in candids}

            if "extra" not in record or record["extra"] is None:
                record["extra"] = {self.__class__.__name__: cutouts}
            else:
                record["extra"][self.__class__.__name__] = cutouts
