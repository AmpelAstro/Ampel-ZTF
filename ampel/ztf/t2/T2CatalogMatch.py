#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/t2/T2CatalogMatch.py
# License:             BSD-3-Clause
# Author:              matteo.giomi@desy.de
# Date:                24.08.2018
# Last Modified Date:  29.01.2021
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>

from typing import Any, Literal, ClassVar
from collections.abc import Sequence
from ampel.types import UBson
from ampel.abstract.AbsPointT2Unit import AbsPointT2Unit
from ampel.content.DataPoint import DataPoint
from ampel.base.AmpelBaseModel import AmpelBaseModel
from ampel.struct.UnitResult import UnitResult
from ampel.ztf.base.CatalogMatchUnit import CatalogMatchUnit
from ampel.enum.DocumentCode import DocumentCode
from ampel.model.DPSelection import DPSelection


class CatalogModel(AmpelBaseModel):
    """
    :param use: either extcats or catsHTM, depending on how the catalog is set up.
    :param rs_arcsec: search radius for the cone search, in arcseconds
    :param catq_kwargs: parameter passed to the catalog query routine.

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

    use: Literal["extcats", "catsHTM"]
    rs_arcsec: float
    keys_to_append: None | Sequence[str]
    pre_filter: None | dict[str, Any]
    post_filter: None | dict[str, Any]
    #: return all matches, not only the closest
    all: bool = False


class T2CatalogMatch(CatalogMatchUnit, AbsPointT2Unit):
    """
    Cross matches the position of a transient to those of sources in a set of catalogs
    """

    # run only on first datapoint by default
    eligible: ClassVar[DPSelection] = DPSelection(
        filter="PPSFilter", sort="jd", select="first"
    )

    # Each value specifies a catalog in extcats or catsHTM format and the query parameters
    catalogs: dict[str, CatalogModel]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.all_catalogs = {k: v for k, v in self.catalogs.items() if v.all}
        self.closest_catalogs = {k: v for k, v in self.catalogs.items() if not v.all}

    def process(self, datapoint: DataPoint) -> UBson | UnitResult:
        """
        :returns: example of a match in SDSS but not in NED:

        {
            'SDSS_spec': {
                'z': 0.08820018172264099,
                'bptclass': 2.0,
                'subclass': '',
                'dist2transient': 1.841666956181802e-09}
            },
            'NED': None
        }

        Note that, when a match is found, the distance of the lightcurve object
        to the catalog counterpart is also returned as the 'dist2transient' key.
        """

        try:
            transient_ra = datapoint["body"]["ra"]
            transient_dec = datapoint["body"]["dec"]
        except KeyError:
            return UnitResult(code=DocumentCode.T2_MISSING_INFO)

        closest_matches = self.cone_search_nearest(
            ra=transient_ra,
            dec=transient_dec,
            catalogs=[
                {
                    "name": catalog,
                    "use": cat_opts.use,
                    "rs_arcsec": cat_opts.rs_arcsec,
                    "keys_to_append": cat_opts.keys_to_append,
                    "pre_filter": cat_opts.pre_filter,
                    "post_filter": cat_opts.post_filter,
                }
                for catalog, cat_opts in self.closest_catalogs.items()
            ],
        )

        all_matches = self.cone_search_all(
            ra=transient_ra,
            dec=transient_dec,
            catalogs=[
                {
                    "name": catalog,
                    "use": cat_opts.use,
                    "rs_arcsec": cat_opts.rs_arcsec,
                    "keys_to_append": cat_opts.keys_to_append,
                    "pre_filter": cat_opts.pre_filter,
                    "post_filter": cat_opts.post_filter,
                }
                for catalog, cat_opts in self.all_catalogs.items()
            ],
        )

        # return the info as dictionary
        body: dict[str, Any] = {
            catalog: {"dist2transient": match["dist_arcsec"], **match["body"]}
            if match is not None
            else None
            for catalog, match in zip(self.closest_catalogs, closest_matches)
        }
        body.update(
            {
                catalog: [
                    {"dist2transient": item["dist_arcsec"], **item["body"]}
                    for item in match
                ]
                if match
                else None
                for catalog, match in zip(self.all_catalogs, all_matches)
            }
        )

        return body
