#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/alert/BTSForcedPhotometryAlertSupplier.py
# License:             BSD-3-Clause
# Author:              valery brinnel
# Date:                25.10.2021
# Last Modified Date:  15.11.2024
# Last Modified By:    jno

import sys
from hashlib import blake2b

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
from bson import encode

from ampel.alert.AmpelAlert import AmpelAlert
from ampel.alert.BaseAlertSupplier import BaseAlertSupplier
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.view.ReadOnlyDict import ReadOnlyDict
from ampel.ztf.alert.calibrate_fps_fork import search_peak
from ampel.ztf.util.ZTFIdMapper import to_ampel_id

dcast = {
    "jd": float,
    "fnu_microJy": float,
    "fnu_microJy_unc": float,
    "passband": str,
    "programid": int,
    "fcqfid": int,
    "zpdiff": float,
    "sys_unc_factor": float,
    "flags": int,
}

ZTF_FILTER_MAP = {"ZTF_g": 1, "ZTF_r": 2, "ZTF_i": 3}


class BTSForcedPhotometryAlertSupplier(BaseAlertSupplier):
    """
    Returns an AmpelAlert instance for each file path provided by the underlying alert loader.

    This unit assuming that input files are *baseline corrected* IPAC
    forced photometry files, of the kind produced by the BTS group.

    These are assumed to be named as:
    {ZTFID}_fnu.csv
    with columns ['jd', 'fnu_microJy', 'fnu_microJy_unc', 'passband', 'fcqfid', 'zpdiff', 'sys_unc_factor', 'flags']


    """

    # FP read and baseline correction
    flux_key: str = "fnu_microJy"
    flux_unc_key: str = "fnu_microJy_unc"
    flux_unc_scale: dict[str, float] = {"ZTF_g": 1.0, "ZTF_r": 1.0, "ZTF_i": 1.0}
    flux_unc_floor: float = 0.02
    baseline_flag_cut: int = (
        512  # This will allow also cases with underestimated scaled unc
    )
    days_prepeak: None | float = (
        None  # Cut epochs earlier than this relative to peak. Assumes this can be found!
    )
    days_postpeak: None | float = (
        None  # Cut epochs later than this relative to peak. Assumes this can be found!
    )
    allow_iband_peak: bool = (
        False  # Include I-band in peak calculation (Warning: I-band incomplete)
    )

    # Candidate coordinates
    # BTS IPAC FP files are named according to a transient, but do not contain coordinates.
    # You can provide a file containing {ZTFid} to {ra,dec} maps, otherwise the file names will be used.
    name_file: str | None = None
    file_keys: dict[str, str] = {
        "id": "ZTFID",
        "ra": "RA",
        "dec": "Dec",
        "raunit": "hourangle",
        "decunit": "deg",
    }  # Name column in file
    name_coordinates = None  # Loaded at first run
    name_values = None  # Loaded at first run

    # Run mode (both can be used)
    alert_history: bool = False  # Return an "alert" for each significant detection
    full_history: bool = True  # Return the full lightcurve, after alerts.
    detection_threshold: float = (
        5.0  # S/N above this considered a significant detection
    )

    # Loaded transient datapoints (for alert mode)
    # Dynamically updated during
    transient_name: str = ""
    transient_tags: list = []
    transient_pps: list = []
    transient_baselineinfo: dict = {}
    transient_hashid: list = []
    alert_counter: int = 0

    def __init__(self, **kwargs) -> None:
        kwargs["deserialize"] = None
        super().__init__(**kwargs)

    def _load_pps(self, fpath: str) -> bool:
        """
        Load BTS IPAC FP datapoints from input file.
        Will set the transient id, tags and pps variables.
        Return False in case nothing was found.
        """

        # post_init did not work for Supplier kind of units, loading name maps here.
        if self.name_file and not self.name_coordinates:
            # Move to init when configured correctly
            df = pd.read_csv(self.name_file)
            self.name_coordinates = SkyCoord(
                ra=df[self.file_keys["ra"]],
                dec=df[self.file_keys["dec"]],
                unit=(self.file_keys["raunit"], self.file_keys["decunit"]),
            )
            self.name_values = df[self.file_keys["id"]]

        # Assuming that filenames follow the BTS convention
        ztfname = fpath.split("/")[-1].split("_")[0]

        # Find coordinates - if name file exists
        ra, dec = None, None  # Allowing events with no coordinate info
        if self.name_coordinates:
            myc = self.name_coordinates[self.name_values == ztfname]
            if len(myc) > 0:
                ra = myc.ra.deg[0]
                dec = myc.dec.deg[0]

        # Read file
        df = pd.read_csv(fpath)
        # Reject data with flags above threshold
        df = df[df["flags"] <= self.baseline_flag_cut]
        tags: list[str] = []

        # Search each unique filter/quadrant combo for a peak, in order to cut outliers
        if self.days_prepeak or self.days_postpeak:
            # Reset index to allow mediam time-search
            # obs_jd = Time(df.jd.values, format="jd")
            df = df.set_index(pd.to_datetime(Time(df.jd.values, format="jd").datetime))
            allpeaks = {}
            for obs_group, df_group in df.groupby("fcqfid"):
                # Skip secondary grid
                if obs_group > 10000000:
                    continue
                if not self.allow_iband_peak and str(obs_group)[-1] == "3":
                    continue
                allpeaks[str(obs_group)] = search_peak(
                    df_group.fnu_microJy,
                    df_group.fnu_microJy_unc,
                    df_group.jd,
                    window="14D",
                )
            peaktimes = [v["t_fcqfid_max"] for k, v in allpeaks.items() if v["det_sn"]]
            if len(peaktimes) == 0:
                print(
                    "No peaks found in lightcurve, but a cut around this was required - skip object."
                )
                return False
            if np.std(peaktimes, ddof=1) > 50:
                print("Warning! Large scatter in time of maximum")
                df["flags"] += 1
            t_peak = np.mean(peaktimes)
            # Could alternatively have weighted with the detection S/N
            # snrs = [v["det_snr"] for k, v in allpeaks.items() if v["det_sn"]]
            # wt_peak = np.average(peaktimes, weights=snrs)
            # print(f"weighted peak time {wt_peak}")
            if self.days_prepeak:
                df = df[(df["jd"] - t_peak) >= -self.days_prepeak]
            if self.days_postpeak:
                df = df[(df["jd"] - t_peak) <= self.days_postpeak]

        pps = []
        alert_ids: list[bytes] = []
        for _, row in df.iterrows():
            pp = {
                str(k): dcast[str(k)](v) if (k in dcast and v is not None) else v
                for k, v in row.items()
            }

            pp["fid"] = ZTF_FILTER_MAP[pp["passband"]]
            pp["ra"] = ra
            pp["dec"] = dec
            pp["rcid"] = (
                (int(str(pp["fcqfid"])[-3]) - 1) * 4 + int(str(pp["fcqfid"])[-2]) - 1
            )

            # Some entries have flux values exactly at 0. Suspicious - we will skip these.
            if pp[self.flux_key] == 0:
                continue

            # Convert jansky to flux
            pp["flux"] = pp[self.flux_key] * 2.75406

            # Opionally scale uncertainties
            pp["flux_unc"] = (
                pp[self.flux_unc_key] * 2.75406 * self.flux_unc_scale[pp["passband"]]
            )

            # Enforce error floor
            if pp["flux_unc"] / abs(pp["flux"]) < self.flux_unc_floor:
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

        self.transient_name = ztfname
        self.transient_tags = tags
        self.transient_pps = pps
        self.transient_hashid = alert_ids
        self.alert_counter = 0
        return True

    def _build_alert(self, datapoints: int) -> AmpelAlertProtocol:
        return AmpelAlert(
            id=int.from_bytes(  # alert id
                blake2b(self.transient_hashid[datapoints - 1], digest_size=7).digest(),
                byteorder=sys.byteorder,
            ),
            stock=to_ampel_id(self.transient_name),  # internal ampel id
            # stock=self.transient_name,  # internal ampel id - for forced photometry we still have not associated to unique ZTF ... do we need to do that later?
            # Ampel alert structure assumes most recent detection to come first
            datapoints=tuple(
                [self.transient_pps[datapoints - 1]]
                + self.transient_pps[0 : datapoints - 1]
            ),
            extra=ReadOnlyDict(
                {
                    "name": self.transient_name,
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
            self._load_pps(
                str(next(self.alert_loader))
            )  # Supplier assumings IO comes as path to file

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
            self.alert_counter = len(self.transient_pps)
            return self._build_alert(self.alert_counter)

        return self.__next__()
