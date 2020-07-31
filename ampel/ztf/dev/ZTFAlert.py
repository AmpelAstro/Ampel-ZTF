#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/dev/ZTFAlert.py
# License           : BSD-3-Clause
# Author            : vb <vbrinnel@physik.hu-berlin.de>
# Date              : 24.06.2018
# Last Modified Date: 31.07.2020
# Last Modified By  : vb <vbrinnel@physik.hu-berlin.de>

import fastavro, os, time
from typing import Dict, Optional, List
from ampel.view.LightCurve import LightCurve
from ampel.view.TransientView import TransientView
from ampel.content.DataPoint import DataPoint
from ampel.alert.PhotoAlert import PhotoAlert
from ampel.ztf.alert.ZiAlertSupplier import ZiAlertSupplier


class ZTFAlert:


	@classmethod
	def to_photo_alert(cls, file_path: Optional[str] = None) -> PhotoAlert:
		"""
		Creates and returns an instance of ampel.view.LightCurve using a ZTF IPAC alert.
		"""
		als = ZiAlertSupplier(deserialize="avro")
		if file_path:
			from ampel.alert.load.FileAlertLoader import FileAlertLoader
			als.set_alert_source(FileAlertLoader(file_path))

		return next(als)


	@classmethod
	def to_lightcurve(cls, file_path: Optional[str] = None) -> LightCurve:
		"""
		Creates and returns an instance of ampel.view.LightCurve using a ZTF IPAC alert.
		"""

		pal = cls.to_photo_alert(file_path)
		photo_shaper = ZIPhotoDataShaper()

		# Build upper limit ids (done by ingester for now)
		if pal.uls:
			for el in pal.uls:
				el['_id'] = int("%i%s%i" % (
					(2457754.5 - el['jd']) * 1000000,
					str(el['pid'])[8:10],
					round(abs(el['diffmaglim']) * 1000)
				))

		uls = photo_shaper.ampelize(
			shaped_content['uls'],
			set(el['_id'] for el in shaped_content['uls']),
			id_field_name='_id'
		)

		for ul in uls:
			ul['stock'] = shaped_content['stockId']

		pps = photo_shaper.ampelize(
			shaped_content['pps'],
			set(el['candid'] for el in shaped_content['pps'])
		)

		for pp in pps:
			pp['stock'] = shaped_content['stockId']

		return LightCurve(
			os.urandom(16), # CompoundI
			tuple([DataPoint(**el) for el in pps]), # Photopoints
			tuple([DataPoint(**el) for el in uls]), # Upperlimit
			0, # tier
			time.time() # added
		)


	# TODO: incomplete/meaningless/quick'n'dirty method, to improve if need be
	@classmethod
	def to_transientview(cls,
		file_path: Optional[str] = None,
		content: Optional[Dict] = None,
		t2_records: Optional[List[Dict]] = None
	) -> TransientView:
		"""
		Note: incomplete/meaningless//quick'n'dirty method, to improve if need be.
		Creates and returns an instance of ampel.view.LightCurve using a ZTF IPAC alert.
		"""

		shaped_content = cls._first_shape(file_path, content)
		lc = cls.to_lightcurve(file_path, content)

		return TransientView(
			stock_id = shaped_content['stockId'],
			tran_names = (shaped_content['ZTFName'],), # tran_names
			tags = None,
			journal = None,
			datapoints = lc.photopoints + lc.upperlimits,
			compounds = None,
			t2records = t2_records,
			channel = None
		)


	@classmethod
	def _load_alert(cls, file_path: str) -> Optional[Dict]:
		""" """
		with open(file_path, 'rb') as f:
			content = cls._deserialize(f)
		return content


	@staticmethod
	def _deserialize(f) -> Optional[Dict]:
		""" """
		reader = fastavro.reader(f)
		return next(reader, None)
