#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/alert/ZiTaggedAlertSupplier.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                04.10.2021
# Last Modified Date:  24.11.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from os.path import basename
from typing import Literal

from ampel.abstract.AbsAlertLoader import AbsAlertLoader
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.alert.BaseAlertSupplier import BaseAlertSupplier
from ampel.alert.load.DirFileNamesLoader import DirFileNamesLoader
from ampel.ztf.alert.ZiAlertSupplier import ZiAlertSupplier


class ZiTaggedAlertSupplier(BaseAlertSupplier):
	"""
	Loads alerts from a (flat) directory.
	Tags potentially embedded in file names will included in alerts.

	For example:
	file ZTFabcdef.91T.TNS.BTS.json:
	  __next__() will return a AmpelAlert instance with tags [91T, TNS, BTS]

	file ZTFabcdef.json:
	  __next__() will return a AmpelAlert instance with no tags

	Note that this supplier is only compatible with DirFileNamesLoader
	"""

	# Override default
	deserialize: None | Literal["avro", "json"] = "avro"
	binary_mode: bool = True


	def __init__(self, **kwargs) -> None:

		super().__init__(**kwargs)

		if not isinstance(self.alert_loader, DirFileNamesLoader):
			raise NotImplementedError("ZiTaggedAlertSupplier only supports DirFileNamesLoader for now")

		# quick n dirty mypy cast
		self.alert_loader: AbsAlertLoader[str] = self.alert_loader # type: ignore
		self.open_mode = "rb" if self.binary_mode else "r"


	def __next__(self) -> AmpelAlertProtocol:
		"""
		:raises StopIteration: when alert_loader dries out.
		"""
		fpath = next(self.alert_loader)

		# basename("/usr/local/auth.AAA.BBB.py").split(".")[1:-1] -> ['AAA', 'BBB']
		base = basename(fpath).split(".")

		with open(fpath, self.open_mode) as alert_file:
			return ZiAlertSupplier.shape_alert_dict(
				self._deserialize(alert_file),
				None if len(base) == 1 else base[1:-1] # type: ignore[arg-type]
			)
