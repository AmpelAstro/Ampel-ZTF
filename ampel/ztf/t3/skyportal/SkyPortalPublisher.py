#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/t3/skyportal/SkyPortalPublisher.py
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                16.09.2020
# Last Modified Date:  16.09.2020
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>

from collections.abc import Generator
from functools import partial
from typing import TYPE_CHECKING, Annotated, Literal

from annotated_types import MinLen

from ampel.abstract.AbsPhotoT3Unit import AbsPhotoT3Unit
from ampel.struct.JournalAttributes import JournalAttributes
from ampel.ztf.t3.skyportal.SkyPortalClient import BaseSkyPortalPublisher

if TYPE_CHECKING:
    from ampel.content.JournalRecord import JournalRecord
    from ampel.struct.T3Store import T3Store
    from ampel.view.TransientView import TransientView


class SkyPortalPublisher(BaseSkyPortalPublisher, AbsPhotoT3Unit):
    #: Save sources to these groups
    groups: Annotated[list[str], MinLen(1)]
    #: Save as candidates for these filters. If None, save for filters
    # "AMPEL.{channel}". If a list, save for these filters. If an empty list,
    # post no candidates.
    filters: None | list[str] = None
    #: Post T2 results as annotations instead of comments
    annotate: bool = False
    #: Explicitly post photometry for each stock. If False, rely on some backend
    #: service (like Kowalski on Fritz) to fill in photometry for sources.
    include_photometry: Literal[False] = False
    cutouts: Literal[None] = None

    process_name: None | str = None

    def process(
        self,
        tviews: Generator["TransientView", JournalAttributes, None],
        t3s: "None | T3Store" = None,
    ) -> None:
        """Pass each view to :meth:`post_candidate`."""
        for view in tviews:
            if self.requires_update(view):
                self.post_source(
                    view,
                    groups=self.groups,
                    annotate=self.annotate,
                )
                if self.filters is not None and self.filters:
                    self.post_candidate(
                        view,
                        filters=self.filters,
                    )

    def _filter_journal_entries(self, jentry: "JournalRecord", after: float):
        """Select journal entries from SkyPortalPublisher newer than last update"""
        return (
            jentry["unit"] == "SkyPortalPublisher"
            and (self.process_name is None or jentry["process"] == self.process_name)
            and jentry["ts"] >= after
        )

    def requires_update(self, view: "TransientView") -> bool:
        # find latest activity activity at lower tiers
        latest_activity = max(
            (
                jentry["ts"]
                for jentry in view.get_journal_entries() or []
                if jentry.get("tier") in {0, 1, 2}
            ),
            default=float("inf"),
        )
        return view.stock is not None and bool(
            view.get_journal_entries(
                tier=3,
                filter_func=partial(
                    self._filter_journal_entries, after=latest_activity
                ),
            )
        )
