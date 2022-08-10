#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/alert/ZTFIPACForcedPhotometryAlertSupplier.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                25.10.2021
# Last Modified Date:  24.11.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

import sys
import pandas as pd
import matplotlib.pyplot as plt
from os.path import basename
from bson import encode
from hashlib import blake2b
from bts_phot.calibrate_fps import get_baseline # type: ignore[import]

from ampel.ztf.util.ZTFIdMapper import to_ampel_id
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.view.ReadOnlyDict import ReadOnlyDict
from ampel.alert.AmpelAlert import AmpelAlert
from ampel.alert.BaseAlertSupplier import BaseAlertSupplier
from ampel.model.PlotProperties import PlotProperties
from ampel.plot.create import create_plot_record

dcast = {
	'field': int,
	'ccdid': int,
	'qid': int,
	'filter': str,
	'pid': int,
	'infobitssci': int,
	'sciinpseeing': float,
	'scibckgnd': float,
	'scisigpix': float,
	'zpmaginpsci': float,
	'zpmaginpsciunc': float,
	'zpmaginpscirms': float,
	'clrcoeff': float,
	'clrcoeffunc': float,
	'ncalmatches': int,
	'exptime': float,
	'adpctdif1': float,
	'adpctdif2': float,
	'diffmaglim': float,
	'zpdiff': float,
	'programid': int,
	'jd': float,
	'rfid': int,
	'forcediffimflux': float,
	'forcediffimfluxunc': float,
	'forcediffimsnr': float,
	'forcediffimchisq': float,
	'forcediffimfluxap': float,
	'forcediffimfluxuncap': float,
	'forcediffimsnrap': float,
	'aperturecorr': float,
	'dnearestrefsrc': float,
	'nearestrefmag': float,
	'nearestrefmagunc': float,
	'nearestrefchi': float,
	'nearestrefsharp': float,
	'refjdstart': float,
	'refjdend': float,
	'procstatus': str,
	'phot_good': bool,
	'flux_standard_corr': float,
	'flux': float,
	'flux_err': float,
	'diffimchisq_corr': float,
	'base': float,
	'base_err': float,
	'SignalNoise_rms': float,
	'name': str,
	'old_stock': int,
	'ra': float,
	'dec': float,
	't_start': float,
	't_end': float,
	'magpsf': float,
	'sigmapsf': float,
	'rcid': int,
	'isdiffpos': str,
	'poor_conditions': int
}

ZTF_FILTER_MAP = {'ZTF_g': 1, 'ZTF_r': 2, 'ZTF_i': 3}


class ZTFIPACForcedPhotometryAlertSupplier(BaseAlertSupplier):
	"""
	Returns an AmpelAlert instance for each file path provided by the underlying alert loader.
	"""

	flux_key: str = "fnu_microJy"
	flux_threshold: int = -20
	flux_unc_key: str = "fnu_microJy_unc"
	flux_unc_scale: dict[str, float] = {'ZTF_g': 1., 'ZTF_r': 1., 'ZTF_i': 1.}
	flux_unc_floor: float = 0.02
	excl_poor_conditions: bool = True

	plot_props: PlotProperties = PlotProperties(
		tags = ["IFP", "BASELINE"],
		file_name = {
			"format_str": "ifp_raw_%s.svg",
			"arg_keys": ["sn_name"]
		},
		title = {
			"format_str": "IFP - %s",
			"arg_keys": ["sn_name"]
		}
	)


	def __init__(self, **kwargs) -> None:

		kwargs['deserialize'] = None
		super().__init__(**kwargs)


	def __next__(self) -> AmpelAlertProtocol:
		"""
		:raises StopIteration: when alert_loader dries out.
		:raises AttributeError: if alert_loader was not set properly before this method is called
		"""

		fpath = next(self.alert_loader) # type: ignore
		with open(fpath, "r") as f: # type: ignore
			li = iter(f)
			for l in li:
				if "# Requested input R.A." in l:
					ra = float(l.split("=")[1].split(" ")[1])
					dec = float(next(li).split("=")[1].split(" ")[1])
					break

		# basename("/usr/local/auth.AAA.BBB.py").split(".")[1:-1] -> ['AAA', 'BBB']
		tags = basename(fpath).split(".")[1:-1] or None # type: ignore
		sn_name = basename(fpath).split(".")[0] # type: ignore

		df = pd.DataFrame()
		fig = plt.figure()
		d = get_baseline(fpath, write_lc=df, make_plot=fig)
		if 't_peak' not in d:
			print(sn_name)
			print(d)
			return self.__next__()

		t_min = d['t_peak'] - 40
		t_max = d['t_peak'] + 150
		all_ids = b""
		pps = []

		for index, row in df.iterrows():

			pp = {
				k: dcast[k](v)
				if (k in dcast and v is not None)
				else v for k, v in row.items()
			}

			if (
				pp['jd'] < t_min or pp['jd'] > t_max or
				(self.excl_poor_conditions and pp['poor_conditions'] == 1) or
				pp[self.flux_key] < self.flux_threshold
			):
				continue

			pp_hash = blake2b(encode(pp), digest_size=7).digest()
			pp['candid'] = int.from_bytes(pp_hash, byteorder=sys.byteorder)
			pp['fid'] = ZTF_FILTER_MAP[pp['passband']]
			pp['ra'] = ra
			pp['dec'] = dec

			# Convert jansky to flux
			pp['flux'] = pp[self.flux_key] * 2.75406

			# Opionally scale uncertainties
			pp['flux_unc'] = pp[self.flux_unc_key] * 2.75406 * self.flux_unc_scale[pp['passband']]

			# Enforce error floor
			if pp['flux_unc'] / pp['flux'] < self.flux_unc_floor:
				if tags is None:
					tags = ['FLOOR']
				else:
					tags.append('FLOOR')
				pp['flux_unc'] = pp['flux'] * self.flux_unc_floor

			all_ids += pp_hash
			pps.append(ReadOnlyDict(pp))

		if not pps:
			return self.__next__()

		pa = AmpelAlert(
			id = int.from_bytes( # alert id
				blake2b(all_ids, digest_size=7).digest(),
				byteorder=sys.byteorder
			),
			stock = to_ampel_id(sn_name), # internal ampel id
			datapoints = tuple(pps),
			extra = ReadOnlyDict(
				{
					'name': sn_name,
					'stock': {
						'ret': d,
						'plot': create_plot_record(
							fig, self.plot_props,
							logger = self.logger,
							extra={"sn_name": sn_name}
						)
					}
				}
			),
			tag = tags
		)

		plt.close('all')
		return pa
