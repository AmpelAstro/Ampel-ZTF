#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/t3/select/T3AdHocStockSelector.py
# License:             BSD-3-Clause
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                17.09.2020
# Last Modified Date:  17.09.2020
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>


from pymongo.cursor import Cursor

from ampel.log.AmpelLogger import AmpelLogger
from ampel.abstract.AbsT3Selector import AbsT3Selector
from ampel.ztf.util.ZTFIdMapper import to_ampel_id


class T3AdHocStockSelector(AbsT3Selector):
    """
    Select specific transients by name. Useful for answering questions from
    astronomers.
    """

    logger: AmpelLogger
    name: list[str]

    def __init__(self, **kwargs):

        if isinstance(name := kwargs.get("name"), str):
            kwargs["name"] = [str]

        super().__init__(**kwargs)

    # Override/Implement
    def fetch(self) -> None | Cursor:
        """ The returned Iterator is a pymongo Cursor """

        cursor = (
            self.context.db.get_collection("stock")
            .find({"_id": {"$in": to_ampel_id(self.name)}}, {"_id": 1})
            .hint("_id_1_channel_1")
        )

        return cursor
