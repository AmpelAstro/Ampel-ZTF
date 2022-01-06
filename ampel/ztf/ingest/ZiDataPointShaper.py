#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/ingest/ZiDataPointShaper.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                14.12.2017
# Last Modified Date:  10.05.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from typing import Any, Iterable, Sequence
from bson import encode
from ampel.util.hash import hash_payload
from ampel.base.AmpelBaseModel import AmpelBaseModel
from ampel.types import StockId, Tag
from ampel.abstract.AbsT0Unit import AbsT0Unit
from ampel.content.DataPoint import DataPoint
from ampel.ztf.ingest.tags import tags

class ZiDataPointShaperBase(AmpelBaseModel):
	"""
	This class 'shapes' datapoints in a format suitable
	to be saved into the ampel database
	"""

	#: JD2017 is used to define upper limits primary IDs
	JD2017: float = 2457754.5
	#: Byte width of datapoint ids
	digest_size: int = 8

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

			base_tags = tags[photo_dict['programid']][photo_dict['fid']]

			is_prv_candidate = popitem(photo_dict, '_is_prv_candidate', False)

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

				# Remove programpi (redundant with programid)
				popitem(photo_dict, 'programpi', None)

				ret_list.append(
					self._create_datapoint(
						stock,
						(["ZTF_ALERT"] if not is_prv_candidate else []) + ["ZTF_DP"],
						photo_dict
					)
				)

			elif "forcediffimflux" in photo_dict:
				
				ret_list.append(self._create_datapoint(stock, ["ZTF_FP"], photo_dict))

			else:

				ret_list.append(
					self._create_datapoint(
						stock,
						["ZTF_UL"],
						{
							'jd': photo_dict['jd'],
							'diffmaglim': photo_dict['diffmaglim'],
							'rcid': (
								rcid
								if (rcid := photo_dict.get('rcid')) is not None
								else (photo_dict['pid'] % 10000) // 100
							),
							'fid': photo_dict['fid'],
							'programid': photo_dict['programid']
						}
					)
				)

		return ret_list
	
	def _create_datapoint(self, stock: StockId, tag: Sequence[Tag], body: dict[str, Any]) -> DataPoint:
		"""
		Create a Datapoint from stock, body, and tags, using the hash of the body as id
		"""
		# ensure that keys are ordered
		sorted_body = dict(sorted(body.items()))
		return {
			"id": hash_payload(encode(sorted_body), size=-self.digest_size*8),
			"stock": stock,
			"tag": [*tags[body['programid']][body['fid']], *tag],
			"body": sorted_body,
		}


class ZiDataPointShaper(ZiDataPointShaperBase, AbsT0Unit):
	
	def process(self, arg: Any, stock: None | StockId = None) -> list[DataPoint]:
		assert stock is not None
		return super().process(arg, stock)
