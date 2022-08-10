#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/alert/ZiAlertSupplier.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                23.04.2018
# Last Modified Date:  24.11.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from typing import Literal, Any
from ampel.types import Tag
from ampel.ztf.util.ZTFIdMapper import to_ampel_id
from ampel.view.ReadOnlyDict import ReadOnlyDict
from ampel.alert.BaseAlertSupplier import BaseAlertSupplier
from ampel.alert.AmpelAlert import AmpelAlert


class ZiAlertSupplier(BaseAlertSupplier):
	"""
	Returns an AmpelAlert instance for each alert payload provided by the underlying alert_loader
	"""

	# Override default
	deserialize: None | Literal["avro", "json"] = "avro"


	def __next__(self) -> AmpelAlert:
		"""
		:raises StopIteration: when alert_loader dries out.
		:raises AttributeError: if alert_loader was not set properly before this method is called
		"""
		d = self._deserialize(
			next(self.alert_loader) # type: ignore
		)

		return self.shape_alert_dict(d)


	@staticmethod
	def shape_alert_dict(
		d: dict[str, Any],
		tag: None | Tag | list[Tag] = None
	) -> AmpelAlert:

		if d['prv_candidates']:

			dps: list[ReadOnlyDict] = [ReadOnlyDict(d['candidate'])]

			for el in d['prv_candidates']:

				# Upperlimit
				if el.get('candid') is None:

					# rarely, meaningless upper limits with negativ
					# diffmaglim are provided by IPAC
					if el['diffmaglim'] < 0:
						continue

					ul = ReadOnlyDict(
						jd = el['jd'],
						fid = el['fid'],
						pid = el['pid'],
						diffmaglim = el['diffmaglim'],
						programid = el['programid'],
						pdiffimfilename = el.get('pdiffimfilename')
					)

					dps.append(ul)

				# PhotoPoint
				else:
					dps.append(ReadOnlyDict(el))

			return AmpelAlert(
				id = d['candid'], # alert id
				stock = to_ampel_id(d['objectId']), # internal ampel id
				datapoints = tuple(dps),
				extra = ReadOnlyDict({'name': d['objectId']}), # ZTF name
				tag = tag
			)

		# No "previous candidate"
		return AmpelAlert(
			id = d['candid'], # alert id
			stock = to_ampel_id(d['objectId']), # internal ampel id
			datapoints = (ReadOnlyDict(d['candidate']), ),
			extra = ReadOnlyDict({'name': d['objectId']}), # ZTF name
			tag = tag
		)
