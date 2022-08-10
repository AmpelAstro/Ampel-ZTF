#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/alert/ZTFGeneralAlertRegister.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                26.05.2020
# Last Modified Date:  24.11.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from struct import pack
from typing import ClassVar, Literal, BinaryIO
from collections.abc import Generator
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.alert.reject.BaseAlertRegister import BaseAlertRegister
from ampel.util.register import reg_iter


class ZTFGeneralAlertRegister(BaseAlertRegister):
	"""
	Saves ZTF stock id with 5 bytes instead of the 8 bytes used by GeneralAlertRegister.
	That is because:
	In []: 2**36 < to_ampel_id('ZTF33zzzzzzz') < 2**37
	Out[]: True
	Logs: alert_id, filter_res, stock
	"""

	__slots__: ClassVar[tuple[str, ...]] = '_write', # type: ignore
	struct: Literal['<QB5s'] = '<QB5s'


	def file(self, alert: AmpelAlertProtocol, filter_res: int = 0) -> None:
		self._write(pack('<QBQ', alert.id, -filter_res, alert.stock)[:-3])


	@classmethod
	def iter(cls,
		f: BinaryIO | str, multiplier: int = 100000, verbose: bool = True
	) -> Generator[tuple[int, ...], None, None]:
		for el in reg_iter(f, multiplier, verbose):
			yield el[0], -el[1], int.from_bytes(el[2], 'little') # type: ignore[arg-type]


	@classmethod
	def find_alert(cls, # type: ignore[override]
		f: BinaryIO | str, alert_id: int | list[int], **kwargs
	) -> None | list[tuple[int, ...]]:
		if ret := super().find_alert(f, alert_id=alert_id, **kwargs):
			return [(el[0], -el[1], int.from_bytes(el[2], 'little')) for el in ret] # type: ignore[arg-type]
		return None


	@classmethod
	def find_stock(cls, # type: ignore[override]
		f: BinaryIO | str, stock_id: int | list[int], **kwargs
	) -> None | list[tuple[int, ...]]:
		if ret := super().find_stock(f, stock_id=stock_id, stock_offset=9, stock_bytes_len=5, **kwargs):
			return [(el[0], -el[1], int.from_bytes(el[2], 'little')) for el in ret] # type: ignore[arg-type]
		return None
