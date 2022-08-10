#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/ingest/ZiDataPointShaper.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                14.12.2017
# Last Modified Date:  10.05.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from typing import Any
from collections.abc import Iterable
from ampel.base.AmpelUnit import AmpelUnit
from ampel.types import StockId
from ampel.abstract.AbsT0Unit import AbsT0Unit
from ampel.content.DataPoint import DataPoint
from ampel.ztf.ingest.tags import tags


class ZiDataPointShaperBase(AmpelUnit):
	"""
	This class 'shapes' datapoints in a format suitable
	to be saved into the ampel database
	"""

	# JD2017 is used to define upper limits primary IDs
	JD2017: float = 2457754.5

	# Mandatory implementation
	def process(self, arg: Iterable[dict[str, Any]], stock: StockId) -> list[DataPoint]: # type: ignore[override]
		"""
		:param arg: sequence of unshaped pps
		IMPORTANT:
		1) This method *modifies* the input dicts (it removes 'candid' and programpi),
		even if the unshaped pps are ReadOnlyDict instances
		2) 'stock' is not set here on purpose since it will conflict with the $addToSet operation
		"""

		ret_list: list[DataPoint] = []
		setitem = dict.__setitem__
		popitem = dict.pop

		for photo_dict in arg:

			# Photopoint
			if photo_dict.get('candid'):

				# Cut path if present
				if photo_dict.get('pdiffimfilename'):
					setitem(
						photo_dict, 'pdiffimfilename',
						photo_dict['pdiffimfilename'] \
							.split('/')[-1] \
							.replace('.fz', '')
					)

				ret_list.append(
					{    # type: ignore[typeddict-item]
						'id': photo_dict['candid'],
						'stock': stock,
						'tag': tags[photo_dict['programid']][photo_dict['fid']],
						'body': photo_dict
					}
				)

				popitem(photo_dict, 'candid', None)
				popitem(photo_dict, 'programpi', None)

			else:

				ret_list.append(
					{    # type: ignore[typeddict-item]
						'id': self.ul_identity(photo_dict),
						'tag': tags[photo_dict['programid']][photo_dict['fid']],
						'stock': stock,
						'body': {
							'jd': photo_dict['jd'],
							'diffmaglim': photo_dict['diffmaglim'],
							'rcid': (
								rcid
								if (rcid := photo_dict.get('rcid')) is not None
								else (photo_dict['pid'] % 10000) // 100
							),
							'fid': photo_dict['fid'],
							'programid': photo_dict['programid']
							#'pdiffimfilename': fname
							#'pid': photo_dict['pid']
						}
					}
				)

		return ret_list


	def ul_identity(self, uld: dict[str, Any]) -> int:
		"""
		Calculate a unique ID for an upper limit from:
		  - jd, floored to the millisecond
		  - readout quadrant number (extracted from pid)
		  - diffmaglim, rounded to 1e-3
		 Example::
		
			>>> ZiT0UpperLimitShaper().identity(
				{
				  'diffmaglim': 19.024799346923828,
				  'fid': 2,
				  'jd': 2458089.7405324,
				  'pdiffimfilename': '/ztf/archive/sci/2017/1202/240532/ztf_20171202240532_000566_zr_c08_o_q1_scimrefdiffimg.fits.fz',
				  'pid': 335240532815,
				  'programid': 0
				}
			)
			-3352405322819025
		"""
		return (
			(int((self.JD2017 - uld['jd']) * 1000000) * 10000000) -
			((rcid if (rcid := uld.get("rcid")) is not None else (uld["pid"] % 10000) // 100) * 100000) -
			round(uld['diffmaglim'] * 1000)
		)

class ZiDataPointShaper(ZiDataPointShaperBase, AbsT0Unit):
	
	def process(self, arg: Any, stock: None | StockId = None) -> list[DataPoint]:
		assert stock is not None
		return super().process(arg, stock)
