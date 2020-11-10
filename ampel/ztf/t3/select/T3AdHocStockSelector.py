#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/t3/select/T3AdHocStockSelector.py
# License           : BSD-3-Clause
# Author            : Jakob van Santen <jakob.van.santen@desy.de>
# Date              : 17.09.2020
# Last Modified Date: 17.09.2020
# Last Modified By  : Jakob van Santen <jakob.van.santen@desy.de>

from typing import Optional, List

from pymongo.cursor import Cursor

from ampel.log.AmpelLogger import AmpelLogger
from ampel.t3.select.AbsT3Selector import AbsT3Selector
from ampel.ztf.utils import to_ampel_id


class T3AdHocStockSelector(AbsT3Selector):
    """
    Select specific transients by name. Useful for answering questions from
    astronomers.
    """

    logger: AmpelLogger
    name: List[str]

    def __init__(self, **kwargs):

        if isinstance(name := kwargs.get("name"), str):
            kwargs["name"] = [str]

        super().__init__(**kwargs)

    # Override/Implement
    def fetch(self) -> Optional[Cursor]:
        """ The returned Iterator is a pymongo Cursor """

        cursor = (
            self.context.db.get_collection("stock")
            .find({"_id": {"$in": to_ampel_id(self.name)}}, {"_id": 1})
            .hint("_id_1_channel_1")
        )

        # Count results
        if cursor.count() == 0:
            self.logger.info(f"No transient named {self.name}")
            return None

        self.logger.info(f"{cursor.count()} transients match search criteria")

        return cursor
