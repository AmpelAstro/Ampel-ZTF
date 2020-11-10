#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : ampel/ztf/t3/complement/FritzReport.py
# License           : BSD-3-Clause
# Author            : Jakob van Santen <jakob.van.santen@desy.de>
# Date              : 03.11.2020
# Date              : 03.11.2020
# Last Modified By  : Jakob van Santen <jakob.van.santen@desy.de>


from functools import cached_property
from typing import Any, Dict, Iterable, Optional, Tuple

from ampel.core.AmpelBuffer import AmpelBuffer
from ampel.model.Secret import Secret
from ampel.t3.complement.AbsT3DataAppender import AbsT3DataAppender

from ampel.ztf.t3.skyportal.SkyPortalClient import SkyPortalClient, SkyPortalAPIError


class FritzReport(SkyPortalClient, AbsT3DataAppender):
    """
    Add source record from SkyPortal
    """

    #: Base URL of SkyPortal server
    base_url: str = "https://fritz.science"
    #: API token
    token: Secret[str] = {"key": "fritz/jno/ampelbot"}  # type: ignore[assignment]

    def get_catalog_item(self, names: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
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
                    if not k in {"thumbnails", "annotations", "groups", "internal_key"}
                }
        return None

    def complement(self, records: Iterable[AmpelBuffer]) -> None:
        for record in records:
            if (stock := record["stock"]) is None:
                raise ValueError(f"{type(self).__name__} requires stock records")
            item = self.get_catalog_item(
                tuple(name for name in (stock["name"] or []) if isinstance(name, str))
            )
            if record.get("extra") is None or record["extra"] is None:
                record["extra"] = {self.__class__.__name__: item}
            else:
                record["extra"][self.__class__.__name__] = item
