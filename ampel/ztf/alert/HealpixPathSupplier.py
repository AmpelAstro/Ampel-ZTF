#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/alert/HealpixPathSupplier.py
# License           : BSD-3-Clause
# Author            : Marcus Fenner <mf@physik.hu-berlin.de>
# Date              : 13.10.2021
# Last Modified Date: 14.06.2022
# Last Modified By  : Marcus Fenner <mf@physik.hu-berlin.de>

import tempfile
from datetime import datetime

import healpy as hp
import numpy as np
import requests
from ampel.log import AmpelLogger
from ampel.ztf.alert.load.ZTFHealpixAlertLoader import HealpixSource
from ampel.ztf.alert.ZiHealpixAlertSupplier import ZiHealpixAlertSupplier
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol


class HealpixPathSupplier(ZiHealpixAlertSupplier):
    """
    Download a Healpix map from a provided URL and process.

    """

    # Process pixels with p-values lower than this limit
    pvalue_limit: float = 0.9

    # URL for healpix retrieval
    map_url: str
    scratch_dir: str  # Local dir where map is saved. File with this name del

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        with tempfile.NamedTemporaryFile(
            prefix="Healpix_", dir=self.scratch_dir, delete=True
        ) as temp:
            self.logger.info(
                "Downloading map",
                extra={"url": self.map_url, "tmpfile": temp.name},
            )
            map_data = requests.get(self.map_url)
            temp.write(map_data.content)

            # Process map
            hpx, headers = hp.read_map(temp.name, h=True, nest=True)
            self.trigger_time = [
                datetime.fromisoformat(header[1])
                for header in headers
                if header[0] == "DATE-OBS"
            ][0]
            self.nside = int(hp.pixelfunc.npix2nside(len(hpx)))

            # Find credible levels
            i = np.flipud(np.argsort(hpx))
            sorted_credible_levels = np.cumsum(hpx[i].astype(float))
            credible_levels = np.empty_like(sorted_credible_levels)
            credible_levels[i] = sorted_credible_levels
            self.healpix_pvalues = list(credible_levels)

            # Create mask for pixel selection
            mask = np.zeros(len(credible_levels), int)
            mask[credible_levels <= self.pvalue_limit] = 1
            ipix = mask.nonzero()[0].tolist()
            self.set_healpix(nside=self.nside, pixels=ipix, time=self.trigger_time)
        return

    def __next__(self) -> AmpelAlertProtocol:
        alert_pvalue = None
        alert = super().__next__()
        if (
            len(
                pos := alert.get_tuples(
                    "ra",
                    "dec",
                    filters=[
                        {
                            "attribute": "magpsf",
                            "operator": "is not",
                            "value": None,
                        }
                    ],
                )
            )
            > 0
        ):
            ra = np.mean([p[0] for p in pos])
            dec = np.mean([p[1] for p in pos])
            theta = 0.5 * np.pi - np.deg2rad(dec)
            phi = np.deg2rad(ra)
            alertpix = hp.pixelfunc.ang2pix(
                hp.npix2nside(len(self.healpix_pvalues)),
                theta,
                phi,
                nest=True,
            )
            alert_pvalue = self.healpix_pvalues[alertpix]

        if alert.extra is not None:
            setitem = dict.__setitem__
            to_add = {
                "pvalue": alert_pvalue,
                "nside": self.nside,
                "time": self.trigger_time,  # .isoformat() if self.trigger_time else None,
            }
            for key, value in to_add.items():
                setitem(alert.extra, key, value)
        return alert
