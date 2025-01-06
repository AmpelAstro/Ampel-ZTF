#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/alert/ZTFIPACForcedPhotometryAlertSupplier.py
# License:             BSD-3-Clause
# Author:              valery brinnel
# Date:                25.10.2021
# Last Modified Date:  15.11.2024
# Last Modified By:    jno

import sys
from hashlib import blake2b
from os.path import basename

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from bson import encode

from ampel.alert.AmpelAlert import AmpelAlert
from ampel.alert.BaseAlertSupplier import BaseAlertSupplier
from ampel.model.PlotProperties import FormatModel, PlotProperties

# from ampel.plot.create import create_plot_record
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.view.ReadOnlyDict import ReadOnlyDict
from ampel.ztf.alert.calibrate_fps_fork import get_baseline
from ampel.ztf.util.ZTFIdMapper import to_ampel_id

dcast = {
    "field": int,
    "ccdid": int,
    "qid": int,
    "filter": str,
    "pid": int,
    "infobitssci": int,
    "sciinpseeing": float,
    "scibckgnd": float,
    "scisigpix": float,
    "zpmaginpsci": float,
    "zpmaginpsciunc": float,
    "zpmaginpscirms": float,
    "clrcoeff": float,
    "clrcoeffunc": float,
    "ncalmatches": int,
    "exptime": float,
    "adpctdif1": float,
    "adpctdif2": float,
    "diffmaglim": float,
    "zpdiff": float,
    "programid": int,
    "jd": float,
    "rfid": int,
    "forcediffimflux": float,
    "forcediffimfluxunc": float,
    "forcediffimsnr": float,
    "forcediffimchisq": float,
    "forcediffimfluxap": float,
    "forcediffimfluxuncap": float,
    "forcediffimsnrap": float,
    "aperturecorr": float,
    "dnearestrefsrc": float,
    "nearestrefmag": float,
    "nearestrefmagunc": float,
    "nearestrefchi": float,
    "nearestrefsharp": float,
    "refjdstart": float,
    "refjdend": float,
    "procstatus": str,
    "phot_good": bool,
    "flux_standard_corr": float,
    "flux": float,
    "flux_err": float,
    "diffimchisq_corr": float,
    "base": float,
    "base_err": float,
    "SignalNoise_rms": float,
    "name": str,
    "old_stock": int,
    "ra": float,
    "dec": float,
    "t_start": float,
    "t_end": float,
    "magpsf": float,
    "sigmapsf": float,
    "rcid": int,
    "isdiffpos": str,
    "poor_conditions": int,
}

ZTF_FILTER_MAP = {"ZTF_g": 1, "ZTF_r": 2, "ZTF_i": 3}


class ZTFIPACForcedPhotometryAlertSupplier(BaseAlertSupplier):
    """
    Returns an AmpelAlert instance for each file path provided by the underlying alert loader.
    """

    # FP read and baseline correction
    flux_key: str = "fnu_microJy"
    flux_unc_key: str = "fnu_microJy_unc"
    flux_unc_scale: dict[str, float] = {"ZTF_g": 1.0, "ZTF_r": 1.0, "ZTF_i": 1.0}
    flux_unc_floor: float = 0.02
    baseline_flag_cut: int = (
        512  # This will allow also cases with underestimated scaled unc
    )
    days_prepeak: float = 40.0
    days_postpeak: float = 150.0

    # Transient naming
    # IPAC FP files do not contain any ZTF names, but are named according to a running index.
    # You can provide a file containing {ra,dec} maps to (ZTF) IDs, otherwise the file names will be used.
    name_file: str | None = None
    name_key: str = "ztfID"  # Name column in file
    name_coordinates = None  # Loaded at first run
    name_values = None  # Loaded at first run
    name_match_radius: float = 1.0  # Search radius to accept name

    # Run mode (both can be used)
    alert_history: bool = False  # Return an "alert" for each significant detection
    full_history: bool = True  # Return the full lightcurve, after alerts.
    detection_threshold: float = (
        5.0  # S/N above this considered a significant detection
    )

    # Loaded transient datapoints (for alert mode)
    # Dynamically updated during
    transient_name: str | int = ""
    transient_tags: list = []
    transient_pps: list = []
    transient_baselineinfo: dict = {}
    transient_hashid: list = []
    alert_counter: int = 0

    plot_props: PlotProperties = PlotProperties(
        tags=["IFP", "BASELINE"],
        file_name=FormatModel(format_str="ifp_raw_%s.svg", arg_keys=["sn_name"]),
        title=FormatModel(format_str="IFP - %s", arg_keys=["sn_name"]),
    )

    def __init__(self, **kwargs) -> None:
        # Not run...
        if self.name_file:
            df = pd.read_csv(self.name_file)
            self.name_coordinates = SkyCoord(
                ra=df["ra"], dec=df["dec"], unit=(u.deg, u.deg)
            )
            self.name_values = df[self.name_key]
        else:
            self.name_coordinates, self.name_values = None, None

        kwargs["deserialize"] = None
        super().__init__(**kwargs)

    def _load_pps(self, fpath: str) -> bool:
        """
        Load IPAC FP datapoints from input file.
        Will set the transient id, tags and pps variables.
        Return False in case nothing was found.
        """

        if self.name_file and not self.name_coordinates:
            # Move to init when configured correctly
            df = pd.read_csv(self.name_file)
            self.name_coordinates = SkyCoord(
                ra=df["ra"], dec=df["dec"], unit=(u.deg, u.deg)
            )
            self.name_values = df[self.name_key]

        with open(fpath) as f:
            li = iter(f)
            for line in li:
                if "# Requested input R.A." in line:
                    ra = float(line.split("=")[1].split(" ")[1])
                    dec = float(next(li).split("=")[1].split(" ")[1])
                    break

        # Parse filename for info
        tags = basename(fpath).split(".")[1:-1] or None
        sn_name: str | int = basename(fpath).split(".")[0]

        df = pd.DataFrame()
        d = get_baseline(
            fpath,
            write_lc=df,
            make_plot=True,
            save_path="/home/jnordin/tmp/bts_phot_fp/",
            save_fig=True,
        )

        # Parse baseline correction for peak estimate
        t_peak = None
        for basedict in d.values():
            if (t_peak := basedict.get("t_peak", None)) is not None:
                break

        if not t_peak:
            print(f"OBS NO PEAK for{fpath} {sn_name}")
            return False

        t_min = t_peak - self.days_prepeak
        t_max = t_peak + self.days_postpeak
        pps = []
        alert_ids: list[bytes] = []

        for _, row in df.iterrows():
            pp = {
                k: dcast[k](v) if (k in dcast and v is not None) else v
                for k, v in row.items()
            }

            if (
                pp["jd"] < t_min
                or pp["jd"] > t_max
                or (self.baseline_flag_cut <= pp["flags"])
            ):
                continue

            pp["fid"] = ZTF_FILTER_MAP[pp["passband"]]
            pp["ra"] = ra
            pp["dec"] = dec
            pp["rcid"] = (pp["ccdid"] - 1) * 4 + pp["qid"] - 1

            # Convert jansky to flux
            pp["flux"] = pp[self.flux_key] * 2.75406

            # Opionally scale uncertainties
            pp["flux_unc"] = (
                pp[self.flux_unc_key] * 2.75406 * self.flux_unc_scale[pp["passband"]]
            )

            # Enforce error floor
            if pp["flux_unc"] / pp["flux"] < self.flux_unc_floor:
                if tags is None:
                    tags = ["FLOOR"]
                else:
                    tags.append("FLOOR")
                pp["flux_unc"] = pp["flux"] * self.flux_unc_floor

            pp_hash = blake2b(encode(pp), digest_size=7).digest()
            #            pp["candid"] = int.from_bytes(pp_hash, byteorder=sys.byteorder)
            pps.append(
                ReadOnlyDict(
                    dict(
                        sorted(pp.items())
                    )  # Ensure ordered keys for duplication search - necessary here?
                )
            )
            # Create a running list of hash ids
            if len(alert_ids) == 0:
                alert_ids = [pp_hash]
            else:
                alert_ids.append(alert_ids[-1] + pp_hash)

        if len(pps) == 0:
            # No datapoints assebled, e.g. if the
            print(f"OBS NO DPS for{fpath} {sn_name}")
            return False

        # Was a list of ZTF names to use supplied?
        if self.name_coordinates:
            c = SkyCoord(ra=pps[0]["ra"], dec=pps[0]["dec"], unit=(u.deg, u.deg))
            idx, d2d, d3d = c.match_to_catalog_sky(self.name_coordinates)
            # print(
            #    "trying to find name", idx, d2d, d2d.to(u.arcsec), self.name_values[idx]
            # )
            if d2d.to(u.arcsec)[0].value < self.name_match_radius:
                sn_name = to_ampel_id(self.name_values[idx])  # type: ignore

        self.transient_name = sn_name
        self.transient_tags = tags  # type: ignore
        self.transient_pps = pps
        self.transient_baselineinfo = d
        self.transient_hashid = alert_ids
        self.alert_counter = 0
        return True

    def _build_alert(self, datapoints: int) -> AmpelAlertProtocol:
        return AmpelAlert(
            id=int.from_bytes(  # alert id
                blake2b(self.transient_hashid[datapoints - 1], digest_size=7).digest(),
                byteorder=sys.byteorder,
            ),
            #            stock=to_ampel_id(self.transient_name),  # internal ampel id
            stock=self.transient_name,  # internal ampel id - for forced photometry we still have not associated to unique ZTF ... do we need to do that later?
            # Ampel alert structure assumes most recent detection to come first
            datapoints=tuple(
                [self.transient_pps[datapoints - 1]]
                + self.transient_pps[0 : datapoints - 1]
            ),
            extra=ReadOnlyDict(
                {
                    "name": self.transient_name,
                    "stock": {
                        "ret": self.transient_baselineinfo,
                    },
                }
            ),
            tag=self.transient_tags,
        )

    def __next__(self) -> AmpelAlertProtocol:
        """
        :raises StopIteration: when alert_loader dries out.
        :raises AttributeError: if alert_loader was not set properly before this method is called
        """

        # Load next lightcurve if we eighter do not have one or already generated an alert with the full
        while len(self.transient_pps) == 0 or self.alert_counter >= len(
            self.transient_pps
        ):
            # Load next lightcurve from file
            # If no lightcurves found, alert loader should raise Stop Iteration
            self._load_pps(next(self.alert_loader))  # type: ignore

        if self.alert_history and self.alert_counter < len(self.transient_pps):
            # Increase counter until a significant detection is found.
            # Set counter to this, return alert with at lightcurve to this point.
            for dp in self.transient_pps[self.alert_counter :]:
                if abs(dp["flux"]) / dp["flux_unc"] > self.detection_threshold:
                    # Detection, stop here
                    self.alert_counter += 1  # still need to nudge this
                    return self._build_alert(self.alert_counter - 1)
                    break
                self.alert_counter += 1

        # Here we either have not generated piecewise alerts at all, or want to make sure we get one final alert with all data
        if self.full_history and self.alert_counter < len(self.transient_pps):
            # Make sure we return full history
            # create - make alert function...
            self.alert_counter = len(self.transient_pps)
            # print(
            #    "yep, need to issue a final alert",
            #    self.alert_counter,
            #    len(self.transient_hashid),
            # )
            return self._build_alert(self.alert_counter)

        return self.__next__()
