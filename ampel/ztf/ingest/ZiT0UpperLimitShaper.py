#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/ingest/ZiT0UpperLimitShaper.py
# License           : BSD-3-Clause
# Author            : vb <vbrinnel@physik.hu-berlin.de>
# Date              : 14.12.2017
# Last Modified Date: 18.03.2020
# Last Modified By  : vb <vbrinnel@physik.hu-berlin.de>

from typing import Dict, List, Any, Iterable, Optional
from ampel.abstract.AbsT0Unit import AbsT0Unit
from ampel.content.DataPoint import DataPoint
from ampel.log.AmpelLogger import AmpelLogger
from ampel.ztf.ingest.tags import tags


class ZiT0UpperLimitShaper(AbsT0Unit):
	"""
	This class 'shapes' upper limits in a format suitable
	to be saved into the ampel database
	"""

	# override
	logger: Optional[AmpelLogger] # type: ignore[assignment]

	# JD2017 is used to defined upper limits primary IDs
	JD2017: float = 2457754.5

	# Mandatory implementation
	def ampelize(self, arg: Iterable[Dict[str, Any]]) -> List[DataPoint]:
		"""
		'stock' (prev. called tranId) is not set here on purpose
		since it would then conflicts with the $addToSet operation
		"""

		return [
			{
				'_id': self.identity(photo_dict),
				'tag': tags[photo_dict['programid']][photo_dict['fid']],
				'body': {
					'jd': photo_dict['jd'],
					'diffmaglim': photo_dict['diffmaglim'],
					'rcid': (photo_dict['pid'] % 10000) // 100,
					'fid': photo_dict['fid']
					#'pdiffimfilename': fname
					#'pid': photo_dict['pid'],
					# programid is contained in alTags
					#'programid': photo_dict['programid'],
				}
			}
			for photo_dict in arg
		]

	def identity(self, uld: Dict[str, Any]) -> int:
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
			})
			-3352405322819025
		"""
		return (
			(int((self.JD2017 - uld['jd']) * 1000000) * 10000000)
			- (((uld["pid"] % 10000)//100) * 100000)
			- round(uld['diffmaglim'] * 1000)
		)
