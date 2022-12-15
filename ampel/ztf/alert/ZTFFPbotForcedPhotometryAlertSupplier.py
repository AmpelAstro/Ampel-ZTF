#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/alert/ZTFFPbotForcedPhotometryAlertSupplier.py
# License           : BSD-3-Clause
# Author            : jno <jnordin@physik.hu-berlin.de>
# Date              : 14.03.2022
# Last Modified Date: 18.08.2022
# Last Modified By  : sr <simeon.reusch@desy.de>

import sys, re, os

from os.path import basename
from bson import encode
from hashlib import blake2b
from typing import Optional, Literal, Union, Any

import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt

# from appdirs import user_cache_dir
from astropy.time import Time
from scipy.stats import median_abs_deviation

# Only works directly on filenames
# from bts_phot.calibrate_fps import get_baseline # type: ignore[import]

from ampel.ztf.util.ZTFIdMapper import to_ampel_id
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.types import Tag
from ampel.view.ReadOnlyDict import ReadOnlyDict
from ampel.alert.AmpelAlert import AmpelAlert
from ampel.alert.BaseAlertSupplier import BaseAlertSupplier
from ampel.model.PlotProperties import PlotProperties


dcast = {
    "sigma": float,
    "sigma.err": float,
    "ampl": float,
    "ampl.err": float,
    "fval": float,
    "chi2": float,
    "chi2dof": float,
    "humidity": float,
    "obsmjd": float,
    "ccdid": int,
    "amp_id": int,
    "gain": float,
    "readnoi": float,
    "darkcur": float,
    "magzp": float,
    "magzpunc": float,
    "magzprms": float,
    "clrcoeff": float,
    "clrcounc": float,
    "zpclrcov": float,
    "zpmed": float,
    "zpavg": float,
    "zprmsall": float,
    "clrmed": float,
    "clravg": float,
    "clrrms": float,
    "qid": int,
    "rcid": int,
    "seeing": float,
    "airmass": float,
    "nmatches": int,
    "maglim": float,
    "status": int,
    "infobits": int,
    "filterid": int,
    "fieldid": int,
    "moonalt": float,
    "moonillf": float,
    "target_x": float,
    "target_y": float,
    "data_hasnan": bool,
    "pass": int,
    "flag": int,
    "cloudy": int,
    "fcqfid": int,
    "baseline": float,
    "baseline_err_mult": float,
    "n_baseline": int,
    "pre_or_post": int,
    "not_baseline": int,
    "ampl_corr": float,
    "ampl_err_corr": float,
    "flux_Jy": float,
    "flux_err_Jy": float,
}


ZTF_FILTER_MAP = {"ZTF_g": 1, "ZTF_r": 2, "ZTF_i": 3}


def get_fpbot_baseline(
    df: pd.DataFrame,
    window: str = "10D",
    min_peak_snr: float = 3,
    risetime: float = 100,
    falltime=365,
    primary_grid_only: bool = False,
    min_det_per_field_band: int = 10,
    zp_max_deviation_from_median: float = 0.5,
    reference_days_before_peak: Optional[float] = 50.0,
) -> pd.DataFrame:
    """
    For each unique baseline combination, estimate and store baseline.
    Partially taken from
    https://github.com/BrightTransientSurvey/ztf_forced_phot/blob/main/bts_phot/calibrate_fps.py.

    risetime (float): days prior to peak to discard from baseline

    falltime ('co'|float): if 'co' this will be estimated from peak mag assuming Co decay, if int this will be taken as direct days.

    primary_grid_only (bool): if this is set to True,
    all observations in the secondary ZTF grid will be
    discarded

    min_det_per_field_band (int): minimum detections required
    per unique combination of field and band

    zp_max_deviation_from_median (float): maximum deviation of the zeropoint from the median zeropoint that is allowed for each obs to survive

    reference_days_before_peak (Opt[float]): number of days the reference images used for a filter/ccd/band-combination have to been made before the estimated peak date (to avoid contamination of the reference images with transient light)

    """
    df["fcqfid"] = np.array(
        df.fieldid.values * 10000
        + df.ccdid.values * 100
        + df.qid.values * 10
        + df.filterid.values
    )

    if primary_grid_only:
        for fid in df.fieldid.unique():
            if len(str(fid)) > 3:
                df = df.query("fieldid != @fid").reset_index(drop=True)

    """
    Remove all combinations of fieldid and filterid where
    minimum detection number per field and band are not met
    (min_det_per_field_band, default 10)
    """
    counts = df.groupby(by=["fieldid", "filterid"]).size().reset_index(name="counts")

    for i, row in counts.iterrows():
        if row["counts"] < min_det_per_field_band:
            fieldid = row["fieldid"]
            filterid = row["filterid"]
            added = fieldid + filterid
            df.query("fieldid + filterid != @added", inplace=True)
    df = df.reset_index(drop=True)

    # Cut datapoints for which magzp deviates too much from median magzp
    median_zp = np.median(df.magzp)
    df["zp_median_deviation"] = np.abs(np.log10(median_zp / df.magzp))
    df.query("zp_median_deviation < @zp_max_deviation_from_median", inplace=True)

    unique_fid = np.unique(df.fcqfid.values).astype(int)

    # Time index for use for rolling window
    df = df.sort_values("obsmjd")
    obs_jd = Time(df["obsmjd"].values, format="mjd")
    df = df.set_index(pd.to_datetime(obs_jd.datetime))

    # Find time of peak in each field/filter/... combo
    fcqfid_dict: dict[str, dict[str, Any]] = {}
    t_peak_list = []
    for ufid in unique_fid:
        fcqfid_dict[str(ufid)] = {}
        this_fcqfid = np.where(df.fcqfid.values == ufid)

        if (ufid % 10 == 3) or (len(this_fcqfid[0]) < 2):
            continue

        fcqf_df = df.iloc[this_fcqfid].copy()
        # Use the pulls from mean to find largest deviation
        pull_series = fcqf_df.ampl / fcqf_df["ampl.err"]
        roll_med = pull_series.rolling(window, center=True).median().values
        # Only use medians with a min nbr of values (otherwise we get edge results)
        t_max = fcqf_df.obsmjd.values[np.argmax(roll_med)]
        #        flux_max = np.max(roll_med)
        flux_max = fcqf_df.ampl.values[np.argmax(roll_med)]
        flux_scatt = median_abs_deviation(fcqf_df.ampl.values, scale="normal")
        peak_snr = flux_max / flux_scatt
        if (peak_snr > min_peak_snr) and (ufid < 10000000):
            fcqfid_dict[str(ufid)]["det_sn"] = True
            fcqfid_dict[str(ufid)]["t_max"] = t_max
            fcqfid_dict[str(ufid)]["flux_max"] = flux_max
            t_peak_list.append(t_max)
        else:
            fcqfid_dict[str(ufid)]["det_sn"] = False

    """
    For all field/ccd/fid combination where we have determined a peak, we can now check if the reference image end date is comfortably prior to the peak and remove this combo if not
    """
    if reference_days_before_peak:

        ufids_to_check = []

        for ufid, res in fcqfid_dict.items():
            if "t_max" in res.keys():
                ufids_to_check.append(ufid)

        if ufids_to_check:

            ref_mjd_dict = get_reference_mjds(fcqfid_list=ufids_to_check)

            for ufid, ref_end_mjd in ref_mjd_dict.items():
                t_max = fcqfid_dict[ufid]["t_max"]

                if (t_max - ref_end_mjd) < reference_days_before_peak:
                    ufid = int(ufid)
                    df.query("fcqfid != @ufid", inplace=True)

    # should we not first convert to a common zeropoint or flux scale (jansky?)
    df["baseline"] = np.zeros_like(df.ampl.values)
    df["baseline_err_mult"] = np.zeros_like(df.ampl.values)
    df["n_baseline"] = np.zeros_like(df.ampl.values).astype(int)
    df["pre_or_post"] = np.zeros_like(df.ampl.values).astype(int)
    df["not_baseline"] = np.zeros_like(df.ampl.values).astype(int)

    if len(t_peak_list) > 0:
        t_peak = np.mean(t_peak_list)
        if len(t_peak_list) > 1 and np.std(t_peak_list, ddof=1) > 10:
            print("Warning! Large scatter in time of maximum")
        fcqfid_dict["t_peak"] = t_peak
        if falltime == "co":
            around_max = np.where(
                (df.obsmjd.values - t_peak > -10) & (df.obsmjd.values - t_peak < 10)
            )
            if len(around_max[0]) > 0:
                diff_flux_around_max = df.ampl.values[around_max]
                mag_min = np.nanmin(
                    df.magzp.values[around_max] - 2.5 * np.log10(diff_flux_around_max)
                )
                # calculate time when SN signal is "gone" via Co56 decay at z ~ 0.09
                t_faded = t_peak + (22.5 - mag_min) / 0.009
            else:
                t_faded = t_peak + 611  # catch strange cases where t_gmax != t_rmax
        elif isinstance(falltime, (float, int)):
            t_faded = t_peak + falltime
        t_risetime = t_peak - risetime
        outside_baseline = np.where(
            (df.obsmjd.values >= t_risetime) & (df.obsmjd.values <= t_faded)
        )
        df.iloc[outside_baseline[0], df.columns.get_loc("not_baseline")] = np.ones(
            len(outside_baseline[0])
        )

        for ufid in unique_fid:
            if ufid % 10 == 4:  # JN: Not sure what this check does
                continue
            else:
                this_fcqfid = np.where(df.fcqfid.values == ufid)
                fcqf_df = df.iloc[this_fcqfid].copy()

                # measure the baseline pre-peak
                pre_bl = np.where((t_peak - fcqf_df.obsmjd.values > 100))
                fcqfid_dict[str(ufid)]["N_pre_peak"] = 0
                if len(pre_bl[0]) > 1:
                    # base_mjd = fcqf_df.obsmjd.values[pre_bl]
                    base_flux = fcqf_df.ampl.values[pre_bl]
                    base_flux_unc = fcqf_df["ampl.err"].values[
                        pre_bl
                    ]  # Would ampl.err work?
                    mask = np.where(
                        np.abs((base_flux - np.median(base_flux)) / base_flux_unc) <= 5
                    )
                    if len(mask[0]) > 1:
                        Cmean = np.average(
                            base_flux[mask], weights=1 / base_flux_unc[mask] ** 2
                        )
                        sum_diff_sq = np.sum(
                            ((base_flux[mask] - Cmean) / (base_flux_unc[mask])) ** 2
                        )
                        chi = 1 / (len(mask[0]) - 1) * sum_diff_sq
                        fcqfid_dict[str(ufid)]["C_pre"] = Cmean
                        fcqfid_dict[str(ufid)]["chi_pre"] = chi
                        fcqfid_dict[str(ufid)]["N_pre_peak"] = len(mask[0])

                # measure the baseline post-peak
                post_bl = np.where((fcqf_df.obsmjd.values > t_faded))
                fcqfid_dict[str(ufid)]["N_post_peak"] = 0
                if len(post_bl[0]) > 1:
                    # local variable 'base_jd' is assigned to but never used
                    # base_jd = fcqf_df.jd.values[post_bl]
                    base_flux = fcqf_df.ampl.values[post_bl]
                    base_flux_unc = fcqf_df["ampl.err"].values[post_bl]
                    mask = np.where(
                        np.abs((base_flux - np.median(base_flux)) / base_flux_unc) <= 5
                    )
                    if len(mask[0]) > 1:
                        Cmean = np.average(
                            base_flux[mask], weights=1 / base_flux_unc[mask] ** 2
                        )
                        sum_diff_sq = np.sum(
                            ((base_flux[mask] - Cmean) / (base_flux_unc[mask])) ** 2
                        )
                        chi = 1 / (len(mask[0]) - 1) * sum_diff_sq
                        fcqfid_dict[str(ufid)]["C_post"] = Cmean
                        fcqfid_dict[str(ufid)]["chi_post"] = chi
                        fcqfid_dict[str(ufid)]["N_post_peak"] = len(mask[0])

                # Decide which baseline to use
                if (fcqfid_dict[str(ufid)]["N_pre_peak"] >= 25) or (
                    (fcqfid_dict[str(ufid)]["N_pre_peak"] > 10)
                    and (fcqfid_dict[str(ufid)]["N_post_peak"] < 25)
                ):
                    df.iloc[
                        this_fcqfid[0], df.columns.get_loc("baseline")
                    ] = fcqfid_dict[str(ufid)]["C_pre"]
                    df.iloc[
                        this_fcqfid[0], df.columns.get_loc("baseline_err_mult")
                    ] = np.ones(len(this_fcqfid[0])) * max(
                        np.sqrt(fcqfid_dict[str(ufid)]["chi_pre"]), 1
                    )
                    df.iloc[
                        this_fcqfid[0], df.columns.get_loc("n_baseline")
                    ] = fcqfid_dict[str(ufid)]["N_pre_peak"]
                    df.iloc[this_fcqfid[0], df.columns.get_loc("pre_or_post")] = -1
                    fcqfid_dict[str(ufid)]["which_baseline"] = "pre"
                elif (fcqfid_dict[str(ufid)]["N_post_peak"] >= 25) or (
                    (fcqfid_dict[str(ufid)]["N_pre_peak"] < 10)
                    and (fcqfid_dict[str(ufid)]["N_post_peak"] >= 25)
                ):
                    df.iloc[
                        this_fcqfid[0], df.columns.get_loc("baseline")
                    ] = fcqfid_dict[str(ufid)]["C_post"]
                    df.iloc[
                        this_fcqfid[0], df.columns.get_loc("baseline_err_mult")
                    ] = np.ones(len(this_fcqfid[0])) * max(
                        np.sqrt(fcqfid_dict[str(ufid)]["chi_post"]), 1
                    )
                    df.iloc[
                        this_fcqfid[0], df.columns.get_loc("n_baseline")
                    ] = fcqfid_dict[str(ufid)]["N_post_peak"]
                    df.iloc[this_fcqfid[0], df.columns.get_loc("pre_or_post")] = 1
                    fcqfid_dict[str(ufid)]["which_baseline"] = "post"
                else:
                    fcqfid_dict[str(ufid)]["which_baseline"] = None

    # Restrict to subset with baseline corrections
    # (These could in principle have been kept in some form)
    df = df[(df["n_baseline"] > 0)]

    df["ampl_corr"] = df["ampl"] - df["baseline"]
    df["ampl_err_corr"] = df["ampl.err"] * df["baseline_err_mult"]

    return df, fcqfid_dict


def get_reference_mjds(fcqfid_list: list) -> dict:
    """
    Get list of references from IPAC and return dates for all unique combinations of fieldid, CCD and filter
    """
    from planobs.utils import get_references

    fieldids = list(
        set([int(str(fcqfid)[: len(str(fcqfid)) - 4]) for fcqfid in fcqfid_list])
    )

    references = get_references(fieldids)

    ref_mjd_dict: dict[int, float] = {}

    for fcqfid in fcqfid_list:
        if len(str(fcqfid)) == 7:
            i = 0
        else:
            i = 1

        fieldid = int(str(fcqfid)[: 3 + i])
        ccdid = int(str(fcqfid)[3 + i : 5 + i])
        qid = int(str(fcqfid)[5 + i : 6 + i])
        fid = int(str(fcqfid)[6 + i : 7 + i])
        _ref = references.query(
            "field == @fieldid and ccdid == @ccdid and qid == @qid and fid == @fid"
        )
        endobsdate = _ref.endobsdate.values[0].split("+")[0]
        endobsdate_mjd = float(Time(endobsdate, format="iso").mjd)
        ref_mjd_dict.update({fcqfid: endobsdate_mjd})

    return ref_mjd_dict


class ZTFFPbotForcedPhotometryAlertSupplier(BaseAlertSupplier):
    """
    Returns an AmpelAlert instance for each file path provided by the underlying alert loader.
    """

    correct_baseline: bool = True

    flux_unc_floor: float = 0.02
    excl_poor_conditions: bool = True
    excl_baseline_pp: bool = False

    pivot_zeropoint: float = 28.0

    transient_risetime: float = 100.0
    transient_falltime: Union[Literal["co"], float] = 365.0

    primary_grid_only: bool = False

    min_det_per_field_band: int = 10

    zp_max_deviation_from_median: float = 0.5

    reference_days_before_peak: Optional[float] = 50.0

    plot_suffix: Optional[str]
    plot_dir: Optional[str]

    save_dir: Optional[str]

    def __init__(self, **kwargs) -> None:

        kwargs["deserialize"] = None
        super().__init__(**kwargs)

    def __next__(self) -> AmpelAlertProtocol:
        """
        :raises StopIteration: when alert_loader dries out.
        :raises AttributeError: if alert_loader was not set properly before this method is called
        """

        fileio = next(self.alert_loader)

        headervals: dict[str, Any] = {
            "name": None,
            "ra": None,
            "dec": None,
            "lastobs": None,
            "lastdownload": None,
            "lastfit": None,
        }

        for byteline in fileio.readlines():
            for headerkey in headervals.keys():
                m = re.search("#(%s)=(.+)" % (headerkey), str(byteline, "UTF-8"))
                if m:
                    headervals[headerkey] = m.group(2)
            still_looking = None in headervals.values()
            if not still_looking:
                break

        name: str = headervals["name"]

        fileio.seek(0)

        tags: list[Tag] = ["FPbot", "ZTF", "ZTF_PRIV"]

        df = pd.read_csv(fileio, sep=",", comment="#")

        if df.shape[0] == 0:
            self.logger.info("Dataframe is empty, skipping")
            return self.__next__()

        if not "pass" in df.keys():
            self.logger.info("No datapoints surviving quality cuts")
            return self.__next__()

        if self.correct_baseline:

            if self.excl_poor_conditions:
                df = df[(df["pass"] == 1)]

            # Correct for common zeropoint
            df["ampl_zp_scale"] = 10 ** ((self.pivot_zeropoint - df["magzp"]) / 2.5)
            df["ampl"] *= df["ampl_zp_scale"]
            df["ampl.err"] *= df["ampl_zp_scale"]

            # Create baseline
            df, baseline_info = get_fpbot_baseline(
                df,
                risetime=self.transient_risetime,
                falltime=self.transient_falltime,
                primary_grid_only=self.primary_grid_only,
                min_det_per_field_band=self.min_det_per_field_band,
                zp_max_deviation_from_median=self.zp_max_deviation_from_median,
                reference_days_before_peak=self.reference_days_before_peak,
            )

            self.logger.info("Corrected baseline", extra=baseline_info)

            if self.save_dir and df.shape[0] > 0:
                outpath = os.path.join(self.save_dir, f"{name}_blcorr.csv")
                with open(outpath, "w") as f:
                    for key, val in headervals.items():
                        f.write(f"#{key}={val}\n")
                    df.to_csv(f)
                self.logger.info(f"Saved baseline to {outpath}")

            if df.shape[0] == 0:
                self.logger.info("No baseline")
                return self.__next__()

            # Plot
            color_dict = {"1": "green", "2": "red", "3": "orange"}

            if self.plot_suffix and self.plot_dir:
                fig, ax = plt.subplots()
                y_max = -99
                for key, binfo in baseline_info.items():
                    if key == "t_peak":
                        continue
                    # if not 'which_baseline' in binfo or not binfo['which_baseline']:
                    #    continue
                    df_sub = df[((df.fcqfid == int(key)) & (df.n_baseline > 0))]
                    if df_sub.shape[0] == 0:
                        continue

                    if "flux_max" in binfo.keys() and binfo["flux_max"] > y_max:
                        y_max = binfo["flux_max"]

                    ax.errorbar(
                        df_sub.obsmjd,
                        df_sub.ampl_corr,
                        df_sub["ampl_err_corr"],
                        fmt="o",
                        mec=color_dict[key[-1]],
                        ecolor=color_dict[key[-1]],
                        mfc="None",
                        alpha=0.7,
                        ms=2,
                        elinewidth=0.8,
                    )

                if y_max == -99:
                    y_max = df.ampl_corr.max()

                peak_times = df[(df["not_baseline"] == 1)].obsmjd

                ax.axhline(y=0, color="0.7", ls="--")
                ax.axvline(x=peak_times.min(), color="0.5", ls="--")
                ax.axvline(x=peak_times.max(), color="0.5", ls="--")
                ax.set_xlabel("Date (MJD)")
                ax.set_ylabel(f"Flux (ZP = {self.pivot_zeropoint} mag)")

                y_min = 10 ** ((self.pivot_zeropoint - 20) / 2.5)
                ax.set_ylim(
                    [-y_min, y_max * 1.4]
                )  # Bottom limit set based on sample runs

                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.plot_dir,
                        "fpbase_%s.%s" % (headervals["name"], self.plot_suffix),
                    )
                )
                plt.close("fig")
                plt.close("all")
                del (fig, ax)
                gc.collect()

            # Add back zp correction (assumed to be used later)
            df["ampl"] /= df["ampl_zp_scale"]
            df["ampl.err"] /= df["ampl_zp_scale"]
            df["ampl_corr"] /= df["ampl_zp_scale"]
            df["ampl_err_corr"] /= df["ampl_zp_scale"]
            F0 = 10 ** (df.magzp / 2.5)
            F0_err = F0 / 2.5 * np.log(10) * df.magzpunc
            Fratio = df.ampl_corr / F0
            Fratio_err = np.sqrt(
                (df.ampl_err_corr / F0) ** 2 + (df.ampl_corr * F0_err / F0**2) ** 2
            )
            df["flux_Jy"] = Fratio
            df["flux_err_Jy"] = Fratio_err

        # First datapoint assumed to be latest_alert
        df.sort_values("obsmjd", ascending=False, inplace=True)

        all_ids = b""
        pps = []
        for index, row in df.iterrows():
            pp = {
                k: dcast[k](v) if (k in dcast and v is not None) else v
                for k, v in row.items()
            }

            if self.excl_baseline_pp and pp["not_baseline"] == 0:
                continue

            if self.correct_baseline:

                if pp["ampl_corr"] > 0:
                    pp["magpsf"] = -2.5 * np.log10(pp["ampl_corr"]) + pp["magzp"]

                    #
                    pp["sigmapsf"] = (
                        1.0857362047581294 * pp["ampl_err_corr"] / pp["ampl_corr"]
                    )

            else:
                if not pp["ampl"] or pp["ampl"] < 0:
                    continue
                pp["magpsf"] = -2.5 * np.log10(pp["ampl"]) + pp["magzp"]
                pp["sigmapsf"] = 1.0857362047581294 * pp["ampl.err"] / pp["ampl"]

            pp["fid"] = pp.pop("filterid")
            pp["jd"] = pp.pop("obsmjd") + 2400000.5
            pp["programid"] = 1
            pp["sigma_err"] = pp.pop("sigma.err")
            pp["ampl_err"] = pp.pop("ampl.err")

            if headervals["ra"] != "None":
                pp["ra"] = float(headervals["ra"])
            else:
                pp["ra"] = None
            if headervals["dec"] != "None":
                pp["dec"] = float(headervals["dec"])
            else:
                pp["dec"] = None

            pp_hash = blake2b(encode(pp), digest_size=7).digest()
            pp["candid"] = int.from_bytes(pp_hash, byteorder=sys.byteorder)
            pps.append(ReadOnlyDict(pp))
            all_ids += pp_hash

        if not pps:
            return self.__next__()

        return AmpelAlert(
            id=int.from_bytes(  # alert id
                blake2b(all_ids, digest_size=7).digest(), byteorder=sys.byteorder
            ),
            stock=to_ampel_id(headervals["name"]),  # internal ampel id
            datapoints=tuple(pps),
            extra={**headervals},
            tag=tags,
        )
