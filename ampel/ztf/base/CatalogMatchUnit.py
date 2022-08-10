#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/base/CatalogMatchUnit.py
# License:             BSD-3-Clause
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                10.03.2021
# Last Modified Date:  10.03.2021
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>

from functools import cached_property
from typing import (
    Sequence,
    Dict,
    Any,
    Literal,
    TypedDict,
    Optional,
    List,
    Union,
    overload,
)

import backoff
import requests
from requests_toolbelt.sessions import BaseUrlSession

from ampel.base.LogicalUnit import LogicalUnit
from ampel.core.ContextUnit import ContextUnit


class BaseConeSearchRequest(TypedDict):
    """
    :param use: either extcats or catsHTM, depending on how the catalog is set up.
    :param rs_arcsec: search radius for the cone search, in arcseconds

    In case 'use' is set to 'extcats', 'catq_kwargs' can (or MUST?) contain the names of the ra and dec
    keys in the catalog (see example below), all valid arguments to extcats.CatalogQuert.findclosest
    can be given, such as pre- and post cone-search query filters can be passed.

    In case 'use' is set to 'catsHTM', 'catq_kwargs' SHOULD contain the the names of the ra and dec
    keys in the catalog if those are different from 'ra' and 'dec' the 'keys_to_append' parameters
    is OPTIONAL and specifies which fields from the catalog should be returned in case of positional match:

    if not present: all the fields in the given catalog will be returned.
    if `list`: just take this subset of fields.

    Example (SDSS_spec):
    {
        'use': 'extcats',
        'catq_kwargs': {
            'ra_key': 'ra',
            'dec_key': 'dec'
        },
        'rs_arcsec': 3,
        'keys_to_append': ['z', 'bptclass', 'subclass']
    }

    Example (NED):
    {
        'use': 'catsHTM',
        'rs_arcsec': 20,
        'keys_to_append': ['fuffa1', 'fuffa2', ..],
    }
    """

    name: str
    use: Literal["extcats", "catsHTM"]
    rs_arcsec: float


class ConeSearchRequest(BaseConeSearchRequest, total=False):
    keys_to_append: None | Sequence[str]
    pre_filter: None | dict[str, Any]
    post_filter: None | dict[str, Any]


class CatalogItem(TypedDict):
    body: dict[str, Any]
    dist_arcsec: float


class CatalogMatchUnitBase:
    """
    A mixin providing catalog matching with catalogmatch-service
    """

    @cached_property
    def session(self) -> BaseUrlSession:
        """
        A session bound to the base URL of the catalogmatch service
        """
        raise NotImplementedError

    @overload
    def _cone_search(
        self,
        method: Literal["any"],
        ra: float,
        dec: float,
        catalogs: Sequence[ConeSearchRequest],
    ) -> list[bool]:
        ...

    @overload
    def _cone_search(
        self,
        method: Literal["nearest"],
        ra: float,
        dec: float,
        catalogs: Sequence[ConeSearchRequest],
    ) -> list[None | CatalogItem]:
        ...

    @overload
    def _cone_search(
        self,
        method: Literal["all"],
        ra: float,
        dec: float,
        catalogs: Sequence[ConeSearchRequest],
    ) -> list[None | list[CatalogItem]]:
        ...

    @backoff.on_exception(
        backoff.expo,
        requests.ConnectionError,
        max_tries=5,
        factor=10,
    )
    @backoff.on_exception(
        backoff.expo,
        requests.HTTPError,
        giveup=lambda e: e.response.status_code not in {503, 504, 429, 408},
        max_time=60,
    )
    def _cone_search(
        self,
        method: Literal["any", "nearest", "all"],
        ra: float,
        dec: float,
        catalogs: Sequence[ConeSearchRequest],
    ) -> list[bool] | list[None | CatalogItem] | list[None | list[CatalogItem]]:
        response = self.session.post(
            f"cone_search/{method}",
            json={
                "ra_deg": ra,
                "dec_deg": dec,
                "catalogs": catalogs,
            },
        )
        response.raise_for_status()
        return response.json()

    def cone_search_any(
        self, ra: float, dec: float, catalogs: Sequence[ConeSearchRequest]
    ) -> list[bool]:
        return self._cone_search("any", ra, dec, catalogs)

    def cone_search_nearest(
        self, ra: float, dec: float, catalogs: Sequence[ConeSearchRequest]
    ) -> list[None | CatalogItem]:
        return self._cone_search("nearest", ra, dec, catalogs)

    def cone_search_all(
        self, ra: float, dec: float, catalogs: Sequence[ConeSearchRequest]
    ) -> list[None | list[CatalogItem]]:
        return self._cone_search("all", ra, dec, catalogs)


class CatalogMatchUnit(CatalogMatchUnitBase, LogicalUnit):
    """
    Catalog matching for LogicalUnits
    """

    require = ("ampel-ztf/catalogmatch",)

    @cached_property
    def session(self) -> BaseUrlSession:
        assert self.resource is not None
        return BaseUrlSession(base_url=self.resource["ampel-ztf/catalogmatch"])


class CatalogMatchContextUnit(CatalogMatchUnitBase, ContextUnit):
    """
    Catalog matching for ContextUnits
    """

    @cached_property
    def session(self) -> BaseUrlSession:
        return BaseUrlSession(
            base_url=self.context.config.get(
                "resource.ampel-ztf/catalogmatch", str, raise_exc=True
            )
        )
