#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/base/CatalogMatchFilter.py
# License:             BSD-3-Clause
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                19.03.2021
# Last Modified Date:  24.11.2021
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>

from typing import Literal, Any, cast

from ampel.abstract.AbsAlertFilter import AbsAlertFilter
from ampel.ztf.base.CatalogMatchUnit import CatalogMatchUnit, ConeSearchRequest
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.model.operator.AnyOf import AnyOf
from ampel.model.operator.AllOf import AllOf
from ampel.base.AmpelBaseModel import AmpelBaseModel


class BaseCatalogMatchRequest(AmpelBaseModel):
    use: Literal["catsHTM", "extcats"]
    name: str
    rs_arcsec: float


class ExtcatsMatchRequest(BaseCatalogMatchRequest):
    use: Literal["extcats"]
    pre_filter: None | dict[str, Any]
    post_filter: None | dict[str, Any]


CatalogMatchRequest = BaseCatalogMatchRequest | ExtcatsMatchRequest


class CatalogMatchFilter(CatalogMatchUnit, AbsAlertFilter):
    """
    A simple filter that matches candidates with a minimum number of previous
    detections (and the most recent detection from a positive subtraction)
    against a set of catalogs. An alert will be accepted if accept condition is
    either None or evaluates to True, and the rejection condition is either not
    or evaluates to False.
    """

    min_ndet: int
    accept: None | CatalogMatchRequest | AnyOf[CatalogMatchRequest] | AllOf[CatalogMatchRequest]
    reject: None | CatalogMatchRequest | AnyOf[CatalogMatchRequest] | AllOf[CatalogMatchRequest]

    # TODO: cache catalog lookups if deeply nested models ever become a thing
    def _evaluate_match(
        self,
        ra: float,
        dec: float,
        selection: CatalogMatchRequest | AnyOf[CatalogMatchRequest] | AllOf[CatalogMatchRequest],
    ) -> bool:
        if isinstance(selection, AllOf):
            return all(
                self.cone_search_any(ra, dec, [cast(ConeSearchRequest, r.dict()) for r in selection.all_of])
            )
        elif isinstance(selection, AnyOf):
            # recurse into OR conditions
            if isinstance(selection.any_of, AllOf):
                return all(self._evaluate_match(ra, dec, clause) for clause in selection.any_of.all_of)
            else:
                return any(
                    self.cone_search_any(ra, dec, [cast(ConeSearchRequest, r.dict()) for r in selection.any_of])
                )
        else:
            return all(self.cone_search_any(ra, dec, [cast(ConeSearchRequest, r.dict()) for r in [selection]]))

    def process(self, alert: AmpelAlertProtocol) -> bool:

        # cut on the number of previous detections
        if len([el for el in alert.datapoints if el['id'] > 0]) < self.min_ndet:
            return False

        # now consider the last photopoint
        latest = alert.datapoints[0]

        # check if it a positive subtraction
        if not (
            latest["isdiffpos"]
            and (latest["isdiffpos"] == "t" or latest["isdiffpos"] == "1")
        ):
            self.logger.debug("rejected: 'isdiffpos' is %s", latest["isdiffpos"])
            return False

        ra = latest["ra"]
        dec = latest["dec"]
        if self.accept:
            if not self._evaluate_match(ra, dec, self.accept):
                return False
        if self.reject:
            if self._evaluate_match(ra, dec, self.reject):
                return False
        return True
