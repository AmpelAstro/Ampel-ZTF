#!/usr/bin/env python
# File:                ampel/ztf/t3/complement/FritzReport.py
# License:             BSD-3-Clause
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                03.11.2020
# Date:                03.11.2020
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>

from collections.abc import Iterable
from typing import Any

from ampel.abstract.AbsBufferComplement import AbsBufferComplement
from ampel.secret.NamedSecret import NamedSecret
from ampel.struct.AmpelBuffer import AmpelBuffer
from ampel.struct.T3Store import T3Store
from ampel.ztf.t3.skyportal.SkyPortalClient import SkyPortalAPIError, SkyPortalClient


class FritzReport(SkyPortalClient, AbsBufferComplement):
    """
    Add source record from SkyPortal
    """

    #: Base URL of SkyPortal server
    base_url: str = "https://fritz.science"
    #: API token
    token: NamedSecret[str] = NamedSecret[str](label="fritz/jno/ampelbot")

    def get_catalog_item(self, names: tuple[str, ...]) -> None | dict[str, Any]:
        """Get catalog entry associated with the stock name"""
        for name in names:
            if name.startswith("ZTF"):
                try:
                    record = self.get(f"sources/{name}")
                except SkyPortalAPIError:
                    return None
                # strip out Fritz chatter
                return {
                    k: v
                    for k, v in record["data"].items()
                    if k not in {"thumbnails", "annotations", "groups", "internal_key"}
                }
        return None

    def update_record(self, record: AmpelBuffer) -> None:
        if (stock := record["stock"]) is None:
            raise ValueError(f"{type(self).__name__} requires stock records")
        item = self.get_catalog_item(
            tuple(name for name in (stock["name"] or []) if isinstance(name, str))
        )
        if record.get("extra") is None or record["extra"] is None:
            record["extra"] = {self.__class__.__name__: item}
        else:
            record["extra"][self.__class__.__name__] = item

    def complement(self, records: Iterable[AmpelBuffer], t3s: T3Store) -> None:
        for record in records:
            self.update_record(record)
