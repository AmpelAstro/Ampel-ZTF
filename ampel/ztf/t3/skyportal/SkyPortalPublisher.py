#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/t3/skyportal/SkyPortalPublisher.py
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                16.09.2020
# Last Modified Date:  16.09.2020
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>

import asyncio, nest_asyncio
from typing import TYPE_CHECKING
from collections.abc import Generator

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.JournalAttributes import JournalAttributes
from ampel.types import StockId
from ampel.ztf.t3.skyportal.SkyPortalClient import BaseSkyPortalPublisher

if TYPE_CHECKING:
    from ampel.view.T3Store import T3Store
    from ampel.view.TransientView import TransientView


class SkyPortalPublisher(BaseSkyPortalPublisher, AbsPhotoT3Unit):

    #: Save sources to these groups
    groups: None | list[str] = None
    filters: None | list[str] = None
    #: Post T2 results as annotations instead of comments
    annotate: bool = False

    def process(self,
        tviews: Generator["TransientView", JournalAttributes, None],
        t3s: 'None | T3Store' = None
    ) -> None:
        """Pass each view to :meth:`post_candidate`."""
        # Patch event loop to be reentrant if it is already running, e.g.
        # within a notebook
        try:
            if asyncio.get_event_loop().is_running():
                nest_asyncio.apply()
        except RuntimeError:
            # second call raises: RuntimeError: There is no current event loop in thread 'MainThread'.
            ...
        asyncio.run(self.post_candidates(tviews))

    async def post_candidates(
        self, tviews: Generator["TransientView", JournalAttributes, None]
    ) -> None:
        """Pass each view to :meth:`post_candidate`."""
        async with self.session(limit_per_host=self.max_parallel_connections):
            await asyncio.gather(
                *[
                    self.post_view(view)
                    for view in tviews
                    if view.stock is not None
                ],
            )

    async def post_view(self, view: "TransientView") -> tuple[StockId, JournalAttributes]:
        return view.id, JournalAttributes(
            extra=dict(
                await self.post_candidate(
                    view,
                    groups=self.groups,
                    filters=self.filters,
                    annotate=self.annotate,
                )
            )
        )
