#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/dev/DevSkyPortalClient.py
# Author            : Jakob van Santen <jakob.van.santen@desy.de>
# Date              : 16.09.2020
# Last Modified Date: 16.09.2020
# Last Modified By  : Jakob van Santen <jakob.van.santen@desy.de>

import gzip
import io
from collections import defaultdict
from datetime import datetime
from typing import Sequence, Dict, Any, Generator

import numpy as np
import requests
from ampel.alert.PhotoAlert import PhotoAlert
from astropy.io import fits
from astropy.time import Time
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


def render_thumbnail(cutout_data: bytes) -> bytes:
    """
    Render gzipped FITS as PNG
    """
    with gzip.open(io.BytesIO(cutout_data), "rb") as f:
        with fits.open(f) as hdu:
            header = hdu[0].header
            img = np.flipud(hdu[0].data)
    mask = np.isfinite(img)

    fig = Figure(figsize=(1, 1))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.imshow(
        img,
        # clip pixel values below the median
        norm=Normalize(*np.percentile(img[mask], [0.5, 99.5])),
        aspect="auto",
        origin="lower",
    )

    with io.BytesIO() as buf:
        fig.savefig(buf, dpi=img.shape[0])
        return buf.getvalue()


class DevSkyPortalClient:
    """
    Post PhotoAlerts to [a local, test instance of] SkyPortal
    """

    def __init__(self, root_token, base_url="http://localhost:9000/api"):
        """
        :param root_token: INITIAL_ADMIN from .tokens.yaml in the SkyPortal container
        """
        self.base_url = base_url
        self.kwargs = {"headers": {"Authorization": f"token {root_token}"}}
        self.session = requests.Session()

        # Set up seed data ourselves
        p48 = self.get_id(
            "/telescope",
            {"name": "P48"},
            {
                "diameter": 1.2,
                "elevation": 1870.0,
                "lat": 33.3633675,
                "lon": -116.8361345,
                "nickname": "Palomar 1.2m Oschin",
                "name": "P48",
                "skycam_link": "http://bianca.palomar.caltech.edu/images/allsky/AllSkyCurrentImage.JPG",
                "robotic": True,
            },
        )

        source = {
            "instrument": self.get_id(
                "/instrument",
                {"name": "ZTF"},
                {
                    "filters": ["ztfg", "ztfr", "ztfi"],
                    "type": "imager",
                    "band": "optical",
                    "telescope_id": p48,
                    "name": "ZTF",
                },
            ),
            "stream": self.get_id("/streams", {"name": "ztf_partnership"}),
            "group": 1,  # root group
        }
        self.post(
            f"/groups/{source['group']}/streams", json={"stream_id": source["stream"]}
        )
        source["filter"] = self.get_id(
            "/filters",
            {"name": "highlander"},
            {
                "name": "highlander",
                "stream_id": source["stream"],
                "group_id": source["group"],
            },
        )
        self.source = source

        # ensure that all users are in the root group
        for user in self.get("/user")["data"]:
            self.post(
                f"/groups/{self.source['group']}/users",
                json={"username": user["username"]},
            )

    def get_id(self, endpoint, params, default=None):
        """Query for an object by id, inserting it if not found"""
        if not (response := self.get(endpoint, params=params))["data"]:
            response = self.post(endpoint, json=default or params, raise_exc=True)
        if isinstance(response["data"], list):
            return response["data"][0]["id"]
        else:
            return response["data"]["id"]

    def request(self, verb, endpoint, raise_exc=False, **kwargs):
        response = self.session.request(
            verb, self.base_url + endpoint, **{**self.kwargs, **kwargs}
        ).json()
        if raise_exc and response["status"] != "success":
            raise RuntimeError(response["message"])
        return response

    def get(self, endpoint, **kwargs):
        return self.request("GET", endpoint, **kwargs)

    def post(self, endpoint, **kwargs):
        return self.request("POST", endpoint, **kwargs)

    def make_photometry(self, alert: PhotoAlert, after=-float("inf")):
        base = {
            "obj_id": alert.id,
            "alert_id": alert.pps[0]["candid"],
            "group_ids": [self.source["group"]],
            "instrument_id": self.source["instrument"],
            "magsys": "ab",
        }
        content = defaultdict(list)
        for doc in self._transform_datapoints(alert.dps, after):
            for k, v in doc.items():
                content[k].append(v)
        return {**base, **content}

    def _transform_datapoints(self, dps: Sequence[Dict[str,Any]], after=-float("inf")) -> Generator[Dict[str,Any],None,None]:
        ztf_filters = {1: "ztfg", 2: "ztfr", 3: "ztfi"}
        for dp in dps:
            if dp["jd"] <= after:
                continue
            base = {
                "filter": ztf_filters[dp["fid"]],
                "mjd": dp["jd"] - 2400000.5,
                "limiting_mag": dp["diffmaglim"],
            }
            if dp["magpsf"] is not None:
                content = {
                    "mag": dp["magpsf"],
                    "magerr": dp["sigmapsf"],
                    "ra": dp["ra"],
                    "dec": dp["dec"],
                }
            else:
                content = {k: None for k in ("mag", "magerr", "ra", "dec")}
            yield {**base, **content}

    def post_alert(self, alert: PhotoAlert):
        # cribbed from https://github.com/dmitryduev/kowalski-dev/blob/882a7fa7e292676dd4864212efa696fb99668b4c/kowalski/alert_watcher_ztf.py#L801-L937
        after = -float("inf")
        if (candidate := self.get(f"/candidates/{alert.id}"))["status"] != "success":
            candidate = alert.pps[0]
            alert_thin = {
                "id": alert.id,
                "ra": candidate.get("ra"),
                "dec": candidate.get("dec"),
                "score": candidate.get("drb", candidate.get("rb")),
                "passing_alert_id": candidate["candid"],
                "filter_ids": [self.source["filter"]],
            }
            self.post("/candidates", json=alert_thin, raise_exc=True)
        elif candidate["data"]["last_detected"]:
            after = Time(datetime.fromisoformat(candidate["data"]["last_detected"])).jd
        # post only if there are new photopoints
        if "mjd" in (photometry := self.make_photometry(alert, after=after)):
            response = self.post("/photometry", json=photometry, raise_exc=True)
