#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/alert/ZTFFPbotForcedPhotometryAlertSupplier.py
# License           : BSD-3-Clause
# Author            : jno <jnordin@physik.hu-berlin.de>
# Date              : 14.03.2022
# Last Modified Date: 14.03.2022
# Last Modified By  : jno <jnordin@physik.hu-berlin.de>

import sys, re, os
import numpy as np
from astropy.time import Time
import pandas as pd
import gc
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
from os.path import basename
from bson import encode
from hashlib import blake2b
from typing import Optional, Literal, Union, Any

# Only works directly on filenames
#from bts_phot.calibrate_fps import get_baseline # type: ignore[import]

from ampel.ztf.util.ZTFIdMapper import to_ampel_id
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.types import Tag
from ampel.view.ReadOnlyDict import ReadOnlyDict
from ampel.alert.AmpelAlert import AmpelAlert
from ampel.alert.BaseAlertSupplier import BaseAlertSupplier
from ampel.model.PlotProperties import PlotProperties



dcast = {
    'sigma': float,
    'sigma.err': float,
    'ampl': float,
    'ampl.err': float,
    'fval': float,
    'chi2': float,
    'chi2dof': float,
    'humidity': float,
    'obsmjd': float,
    'ccdid': int,
    'amp_id': int,
    'gain': float,
    'readnoi': float,
    'darkcur': float,
    'magzp': float,
    'magzpunc': float,
    'magzprms': float,
    'clrcoeff': float,
    'clrcounc': float,
    'zpclrcov': float,
    'zpmed': float,
    'zpavg': float,
    'zprmsall': float,
    'clrmed': float,
    'clravg': float,
    'clrrms': float,
    'qid': int,
    'rcid': int,
    'seeing': float,
    'airmass': float,
    'nmatches': int,
    'maglim': float,
    'status': int,
    'infobits': int,
    'filterid': int,
    'fieldid': int,
    'moonalt': float,
    'moonillf': float,
    'target_x': float,
    'target_y': float,
    'data_hasnan': bool,
    'pass': int,
    'flag': int,
    'cloudy': int,
    'fcqfid': int,
    'baseline': float,
    'baseline_err_mult': float,
    'n_baseline': int,
    'pre_or_post': int,
    'not_baseline': int,
    'ampl_corr': float,
    'ampel_err_corr': float,
}


ZTF_FILTER_MAP = {'ZTF_g': 1, 'ZTF_r': 2, 'ZTF_i': 3}


def get_fpbot_baseline(df: pd.DataFrame, window="10D", min_peak_snr=3,
                       risetime=100, falltime=365) -> pd.DataFrame:
    """
    For each unique baseline combination, estimate and store baseline.
    Partially taken from
	https://github.com/BrightTransientSurvey/ztf_forced_phot/blob/main/bts_phot/calibrate_fps.py.

    risetime (float): days prior to peak to discard from baseline
    falltime ('co'|float): if 'co' this will be estimated from peak mag
        assuming Co decay, if int this will be taken as direct days.
    """

    df['fcqfid'] = np.array(df.fieldid.values*10000 +
	                        df.ccdid.values*100 +
                            df.qid.values*10 +
                            df.filterid.values
                            )
    unique_fid = np.unique(df.fcqfid.values).astype(int)

    # Time index for use for rolling window
    df = df.sort_values('obsmjd')
    obs_jd = Time(df['obsmjd'].values, format='mjd')
    df = df.set_index(pd.to_datetime(obs_jd.datetime))

    # Find time of peak in each field/filter/... combo
    fcqfid_dict: dict[str,dict[str,Any]] = {}
    t_peak_list = []
    for ufid in unique_fid:
        fcqfid_dict[str(ufid)] = {}
        this_fcqfid = np.where(df.fcqfid.values == ufid)

        if ((ufid % 10 == 3) or (len(this_fcqfid[0]) < 2)):
            continue

        fcqf_df = df.iloc[this_fcqfid].copy()
        # Use the pulls from mean to find largest deviation
        pull_series = fcqf_df.ampl / fcqf_df['ampl.err']
        roll_med = pull_series.rolling(window, center=True).median().values
        # Only use medians with a min nbr of values (otherwise we get edge results)
        t_max = fcqf_df.obsmjd.values[np.argmax(roll_med)]
#        flux_max = np.max(roll_med)
        flux_max = fcqf_df.ampl.values[np.argmax(roll_med)]
        flux_scatt = median_abs_deviation(fcqf_df.ampl.values, scale='normal')
        peak_snr = flux_max/flux_scatt
        if ((peak_snr > min_peak_snr) and (ufid < 10000000)):
            fcqfid_dict[str(ufid)]['det_sn'] = True
            fcqfid_dict[str(ufid)]['t_max'] = t_max
            fcqfid_dict[str(ufid)]['flux_max'] = flux_max
            t_peak_list.append(t_max)
        else:
            fcqfid_dict[str(ufid)]['det_sn'] = False

    #should we not first convert to a common zeropoint or flux scale (jansky?)

    df['baseline'] = np.zeros_like(df.ampl.values)
    df['baseline_err_mult'] = np.zeros_like(df.ampl.values)
    df['n_baseline'] = np.zeros_like(df.ampl.values).astype(int)
    df['pre_or_post'] = np.zeros_like(df.ampl.values).astype(int)
    df['not_baseline'] = np.zeros_like(df.ampl.values).astype(int)

    if len(t_peak_list) > 0:
        t_peak = np.mean(t_peak_list)
        if len(t_peak_list) > 1 and np.std(t_peak_list, ddof=1) > 10:
            print('Warning! Large scatter in time of maximum')
        fcqfid_dict['t_peak'] = t_peak
        if falltime == 'co':
            around_max = np.where((df.obsmjd.values - t_peak > - 10) &
			                      (df.obsmjd.values - t_peak < 10))
            if len(around_max[0]) > 0:
                diff_flux_around_max = df.ampl.values[around_max]
                mag_min = np.nanmin(df.magzp.values[around_max] -
                                    2.5*np.log10(diff_flux_around_max))
                #calculate time when SN signal is "gone" via Co56 decay at z ~ 0.09
                t_faded = t_peak + (22.5 - mag_min)/0.009
            else:
                t_faded = t_peak + 611 # catch strange cases where t_gmax != t_rmax
        elif isinstance(falltime, (float, int)):
            t_faded = t_peak + falltime
        t_risetime = t_peak - risetime
        outside_baseline = np.where((df.obsmjd.values >= t_risetime) &
                                    (df.obsmjd.values <= t_faded))
        df.iloc[outside_baseline[0],
                df.columns.get_loc('not_baseline')] = np.ones(len(outside_baseline[0]))



        for ufid in unique_fid:
            if ufid % 10 == 4:    # JN: Not sure what this check does
                continue
            else:
                this_fcqfid = np.where(df.fcqfid.values == ufid)
                fcqf_df = df.iloc[this_fcqfid].copy()

                # measure the baseline pre-peak
                pre_bl = np.where((t_peak - fcqf_df.obsmjd.values > 100))
                fcqfid_dict[str(ufid)]['N_pre_peak'] = 0
                if len(pre_bl[0]) > 1:
                    #base_mjd = fcqf_df.obsmjd.values[pre_bl]
                    base_flux = fcqf_df.ampl.values[pre_bl]
                    base_flux_unc = fcqf_df['ampl.err'].values[pre_bl]  # Would ampl.err work?
                    mask = np.where(np.abs((base_flux - np.median(base_flux))
					                       /base_flux_unc ) <= 5)
                    if len(mask[0]) > 1:
                        Cmean = np.average(base_flux[mask],
                                           weights=1/base_flux_unc[mask]**2)
                        sum_diff_sq = np.sum(((base_flux[mask] - Cmean) /
                                                (base_flux_unc[mask]))**2)
                        chi = 1/(len(mask[0])-1)*sum_diff_sq
                        fcqfid_dict[str(ufid)]['C_pre'] = Cmean
                        fcqfid_dict[str(ufid)]['chi_pre'] = chi
                        fcqfid_dict[str(ufid)]['N_pre_peak'] = len(mask[0])

                # measure the baseline post-peak
                post_bl = np.where((fcqf_df.obsmjd.values > t_faded))
                fcqfid_dict[str(ufid)]['N_post_peak'] = 0
                if len(post_bl[0]) > 1:
                    # local variable 'base_jd' is assigned to but never used
                    #base_jd = fcqf_df.jd.values[post_bl]
                    base_flux = fcqf_df.ampl.values[post_bl]
                    base_flux_unc = fcqf_df['ampl.err'].values[post_bl]
                    mask = np.where(np.abs((base_flux - np.median(base_flux)
                                            )/base_flux_unc )<= 5)
                    if len(mask[0]) > 1:
                        Cmean = np.average(base_flux[mask],
                                            weights=1/base_flux_unc[mask]**2)
                        sum_diff_sq = np.sum(((base_flux[mask] - Cmean) /
                                            (base_flux_unc[mask]))**2)
                        chi = 1/(len(mask[0])-1)*sum_diff_sq
                        fcqfid_dict[str(ufid)]['C_post'] = Cmean
                        fcqfid_dict[str(ufid)]['chi_post'] = chi
                        fcqfid_dict[str(ufid)]['N_post_peak'] = len(mask[0])

                # Decide which baseline to use
                if ((fcqfid_dict[str(ufid)]['N_pre_peak'] >= 25) or
                   ((fcqfid_dict[str(ufid)]['N_pre_peak'] > 10) and
                    (fcqfid_dict[str(ufid)]['N_post_peak'] < 25))
                    ):
                    df.iloc[this_fcqfid[0],
                            df.columns.get_loc('baseline')] = fcqfid_dict[str(ufid)]['C_pre']
                    df.iloc[this_fcqfid[0],
                            df.columns.get_loc('baseline_err_mult')] = np.ones(
                            len(this_fcqfid[0])) * max(np.sqrt(fcqfid_dict[str(ufid)]['chi_pre']),1 )
                    df.iloc[this_fcqfid[0],
                            df.columns.get_loc('n_baseline')] = fcqfid_dict[str(ufid)]['N_pre_peak']
                    df.iloc[this_fcqfid[0],
                        df.columns.get_loc('pre_or_post')] = -1
                    fcqfid_dict[str(ufid)]['which_baseline'] = 'pre'
                elif ((fcqfid_dict[str(ufid)]['N_post_peak'] >= 25) or
                     ((fcqfid_dict[str(ufid)]['N_pre_peak'] < 10) and
                      (fcqfid_dict[str(ufid)]['N_post_peak'] >= 25))
                      ):
                    df.iloc[this_fcqfid[0],
                            df.columns.get_loc('baseline')] = fcqfid_dict[str(ufid)]['C_post']
                    df.iloc[this_fcqfid[0],
                            df.columns.get_loc('baseline_err_mult')] = np.ones(
                        len(this_fcqfid[0])) * max(np.sqrt(fcqfid_dict[str(ufid)]['chi_post']),1 )
                    df.iloc[this_fcqfid[0],
                            df.columns.get_loc('n_baseline')] = fcqfid_dict[str(ufid)]['N_post_peak']
                    df.iloc[this_fcqfid[0],
                            df.columns.get_loc('pre_or_post')] = 1
                    fcqfid_dict[str(ufid)]['which_baseline'] = 'post'
                else:
                    fcqfid_dict[str(ufid)]['which_baseline'] = None

    # Restrict to subset with baseline corrections
    # (These could in principle have been kept in some form)
    df = df[(df['n_baseline']>0)]

    df['ampl_corr'] = df['ampl'] - df['baseline']
    df['ampl_err_corr'] = df['ampl.err'] * df['baseline_err_mult']

    return df, fcqfid_dict



class ZTFFPbotForcedPhotometryAlertSupplier(BaseAlertSupplier):
    """
    Returns an AmpelAlert instance for each file path provided by the underlying alert loader.
    """

    flux_unc_floor: float = 0.02
    excl_poor_conditions: bool = True
    excl_baseline_pp: bool = False

    pivot_zeropoint: float = 28.

    transient_risetime: float = 100.
    transient_falltime: Union[Literal["co"], float] = 365.

    plot_suffix: Optional[str]
    plot_dir: Optional[str]


    def __init__(self, **kwargs) -> None:

        kwargs['deserialize'] = None
        super().__init__(**kwargs)


    def __next__(self) -> AmpelAlertProtocol:
        """
        :raises StopIteration: when alert_loader dries out.
        :raises AttributeError: if alert_loader was not set properly before this method is called
        """

        fileio = next(self.alert_loader)

#        headervals= {}
        headervals: dict[str,Any] = {'ra':None, 'dec':None, 'name': None, 'lastobs':None, 'lastdownload':None, 'lastfit':None}

#        ra, dec, snname, lastobs, lastdownload, lastfit = None, None, None, None, None, None
        for byteline in fileio.readlines():
            for headerkey in headervals.keys():
                m = re.search("#(%s)=(.+)"%(headerkey), str(byteline, 'UTF-8'))
                if m:
                    headervals[headerkey] = m.group(2)
            still_looking = None in headervals.values()
            if not still_looking:
                break

        fileio.seek(0)
        tags: list[Tag] = ['FPbot', 'ZTF', 'ZTF_PRIV']

        df = pd.read_csv(fileio, sep=',', comment='#')

        if self.excl_poor_conditions:
			# Should be equivalent
#            df = df[(df['ampl.err']>0) & (df['chi2dof']<3) & (df['cloudy']==0) & (df['infobits']==0) & (df['pass']==1)]
            df = df[(df['pass'] == 1)]


        # Correct for common zeropoint
        df['ampl_zp_scale'] = 10**((self.pivot_zeropoint - df['magzp']) / 2.5)
        df['ampl'] *= df['ampl_zp_scale']
        df['ampl.err'] *= df['ampl_zp_scale']

        # Create baseline
        df, baseline_info = get_fpbot_baseline(df, risetime=self.transient_risetime, falltime=self.transient_falltime)
        self.logger.info('Corrected baseline', extra=baseline_info)

        if df.shape[0]==0:
            self.logger.info('No baseline')
            return self.__next__()

        # Plot
        color_dict = {'1': 'MediumAquaMarine', '2': 'Crimson', '3': 'Goldenrod'}
        if self.plot_suffix and self.plot_dir:
            fig = plt.figure()
            y_max = -99
            for key, binfo in baseline_info.items():
                if key =='t_peak':
                    continue
                #if not 'which_baseline' in binfo or not binfo['which_baseline']:
                #    continue
                df_sub = df[((df.fcqfid == int(key)) & (df.n_baseline>0))]
                if df_sub.shape[0] == 0:
                    continue

                if 'flux_max' in binfo.keys() and binfo['flux_max'] > y_max:
                    y_max = binfo['flux_max']
                plt.errorbar(df_sub.obsmjd, df_sub.ampl, df_sub['ampl.err'],
                             fmt='^',
                             mec='grey',
                             ecolor=color_dict[key[-1]],
                             mfc='None')
                plt.errorbar(df_sub.obsmjd, df_sub.ampl_corr, df_sub.ampl_err_corr,
                             fmt='o',
                             mec=color_dict[key[-1]],
                             ecolor=color_dict[key[-1]],
                             mfc='None')


            peak_times = df[(df['not_baseline'] == 1)].obsmjd
            plt.axvline(x=peak_times.min(), color='0.5', ls='--')
            plt.axvline(x=peak_times.max(), color='0.5', ls='--')
            plt.ylim([-1000,y_max*2]) # Bottom limit set based on sample runs

            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'fpbase_%s.%s'%(headervals['name'], self.plot_suffix)))
            plt.close('fig')
            plt.close('all')
            del(fig)
            gc.collect()


        # Add back zp correction (assumed to be used later)
        df['ampl'] /= df['ampl_zp_scale']
        df['ampl.err'] /= df['ampl_zp_scale']
        df['ampl_corr'] /= df['ampl_zp_scale']
        df['ampl_err_corr'] /= df['ampl_zp_scale']

        # First datapoint assumed to be latest_alert
        df.sort_values('obsmjd', ascending=False, inplace=True)


        all_ids = b""
        pps = []
        for index, row in df.iterrows():
            pp = {
                k: dcast[k](v)
                   if (k in dcast and v is not None)
                   else v for k, v in row.items()
            }

            if self.excl_baseline_pp and pp['not_baseline'] == 0:
                continue
#            if not pp['ampl'] or pp['ampl'] < 0:
#                continue

            if pp['ampl_corr'] > 0:
                pp['magpsf'] = -2.5 * np.log10(pp['ampl_corr']) + pp['magzp']
                pp['sigmapsf'] = 1.0857362047581294 * pp['ampl_err_corr'] / pp['ampl_corr']

            pp['fid'] = pp.pop('filterid')
            pp['jd'] = pp.pop('obsmjd') + 2400000.5
            pp['programid'] = 1
            pp['sigma_err'] = pp.pop('sigma.err')
            pp['ampl_err'] = pp.pop('ampl.err')
            pp['ra'] = float(headervals['ra'])
            pp['dec'] = float(headervals['dec'])

            pp_hash = blake2b(encode(pp), digest_size=7).digest()
            pp['candid'] = int.from_bytes(pp_hash, byteorder=sys.byteorder)
            pps.append(ReadOnlyDict(pp))
            all_ids += pp_hash

        if not pps:
            return self.__next__()

        return AmpelAlert(
            id=int.from_bytes( # alert id
                blake2b(all_ids, digest_size=7).digest(),
                byteorder=sys.byteorder
            ),
            stock=to_ampel_id(headervals['name']), # internal ampel id
            datapoints=tuple(pps),
            extra={**headervals},
            tag=tags
        )
