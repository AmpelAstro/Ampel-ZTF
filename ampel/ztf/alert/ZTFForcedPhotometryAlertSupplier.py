#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/alert/ZTFForcedPhotometryAlertSupplier.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                29.03.2021
# Last Modified Date:  24.11.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

import sys
from math import log10
from bson import encode
from hashlib import blake2b
from typing import Literal
from os.path import basename
from ampel.ztf.util.ZTFIdMapper import to_ampel_id
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.view.ReadOnlyDict import ReadOnlyDict
from ampel.alert.AmpelAlert import AmpelAlert
from ampel.alert.BaseAlertSupplier import BaseAlertSupplier

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
	'maglim': float,
	'status': int,
	'filterid': int,
	'fieldid': int,
	'moonalt': float,
	'moonillf': float,
	'target_x': float,
	'target_y': float,
	'data_hasnan': bool
}


class ZTFForcedPhotometryAlertSupplier(BaseAlertSupplier):
	"""
	Returns an AmpelAlert instance for each file path provided by the underlying alert loader.
	"""

	dpid: Literal['hash', 'inc'] = 'hash'


	def __init__(self, **kwargs) -> None:

		kwargs['deserialize'] = 'csv'
		super().__init__(**kwargs)
		self.counter = 0 if self.dpid == "hash" else 1


	def __next__(self) -> AmpelAlertProtocol:
		"""
		:raises StopIteration: when alert_loader dries out.
		:raises AttributeError: if alert_loader was not set properly before this method is called
		"""

		fpath = next(self.alert_loader) # type: ignore
		with open(fpath, "r") as fd: # type: ignore
			# Convert first line comment "# key1: val1, key2: val2" into dict (requires loader binary_mode=False)
			cdict = {(x := el.split(":"))[0].strip(): x[1].strip() for el in fd.readline()[1:].split(",")}
			dict_reader = self.deserialize(fd) # type: ignore

		tags = basename(fpath).split(".")[1:-1] or None # type: ignore

		all_ids = b""
		pps = []
		for row in dict_reader:
			pp = {k: dcast[k](v) if len(v) > 0 and k in dcast else v for k, v in row.items()}
			if not pp['ampl'] or pp['ampl'] < 0:
				continue
			pp_hash = blake2b(encode(pp), digest_size=7).digest()
			if self.counter:
				pp['candid'] = self.counter
				self.counter += 1
			else:
				pp['candid'] = int.from_bytes(pp_hash, byteorder=sys.byteorder)
			pp['magpsf'] = -2.5 * log10(pp['ampl']) + pp['magzp']
			# 2.5/log(10) = 1.0857362047581294
			pp['sigmapsf'] = 1.0857362047581294 * pp['ampl.err'] / pp['ampl']
			pp['fid'] = pp.pop('filterid')
			pp['jd'] = pp.pop('obsmjd') + 2400000.5
			pp['programid'] = 1
			pp['sigma_err'] = pp.pop('sigma.err')
			pp['ampl_err'] = pp.pop('ampl.err')
			pp['ra'] = float(cdict['ra'])
			pp['dec'] = float(cdict['dec'])
			all_ids += pp_hash
			pps.append(ReadOnlyDict(pp))

		if not pps:
			return self.__next__()

		return AmpelAlert(
			id = int.from_bytes( # alert id
				blake2b(all_ids, digest_size=7).digest(),
				byteorder=sys.byteorder
			),
			stock = to_ampel_id(cdict['name']), # internal ampel id
			datapoints = tuple(pps),
			extra = {'name': cdict['name']},
			tag = tags
		)
