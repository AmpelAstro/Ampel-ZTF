#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/utils.py
# License           : BSD-3-Clause
# Author            : vb <vbrinnel@physik.hu-berlin.de>
# Date              : 07.06.2018
# Last Modified Date: 27.04.2020
# Last Modified By  : vb <vbrinnel@physik.hu-berlin.de>

from typing import overload, List, Union, Iterable
from ampel.type import StrictIterable, StockId

# Optimization variables
alphabet = "abcdefghijklmnopqrstuvwxyz"
ab = {alphabet[i]: i for i in range(26)}
rg = (6, 5, 4, 3, 2, 1, 0)
powers = tuple(26**i for i in (6, 5, 4, 3, 2, 1, 0))
enc_ztf_years = {str(i + 17): i for i in range(16)}
dec_ztf_years = {i: str(i + 17) for i in range(16)}


@overload
def to_ampel_id(ztf_id: str) -> int:
	...

@overload
def to_ampel_id(ztf_id: StrictIterable[str]) -> List[int]:
	...

def to_ampel_id(ztf_id: Union[str, StrictIterable[str]]) -> Union[int, List[int]]:
	"""
	:returns: ampel id (positive integer).

	====== First 4 bits encode the ZTF year (until max 2032) =====

	In []: to_ampel_id('ZTF17aaaaaaa')
	Out[]: 0

	In []: to_ampel_id('ZTF18aaaaaaa')
	Out[]: 1

	In []: to_ampel_id('ZTF19aaaaaaa')
	Out[]: 2

	====== Bits onwards encode the ZTF name converted from base 26 into base 10 =====

	In []: to_ampel_id('ZTF17aaaaaaa')
	Out[]: 0

	In []: to_ampel_id('ZTF17aaaaaab')
	Out[]: 16

	In []: to_ampel_id('ZTF17aaaaaac')
	Out[]: 32

	====== Biggest numerical value is < 2**37 =====

	Out[]: In []: to_ampel_id('ZTF32zzzzzzz')
	Out[]: 128508962815

	========================================================================
	This encoding allows to save most of ZTF transients (up until ~akzzzzzz)
	with a signed int32. Note: MongoDB imposes signed integers and chooses
	automatically the right _id type (int32/int64/...) per document
	=====================================================================

	In []: to_ampel_id('ZTF20akzzzzzz') < 2**31
	Out[]: True

	In []: to_ampel_id('ZTF20alzzzzzz') < 2**31
	Out[]: False

	=====================================================================

	Note: slightly slower than the legacy method

	-> Legacy
	%timeit to_ampel_id("ZTF19abcdfef")
	949 ns ± 9.08 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

	-> This method
	%timeit to_ampel_id('ZTF19abcdfef')
	1.51 µs ± 35.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

	but encodes ZTF ids with 37 bits instead of 52 bits

	In []: 2**36 < to_ampel_id('ZTF32zzzzzzz') < 2**37
	Out[]: True

	from ampel.ztf.legacy_utils import to_ampel_id as to_legacy_ampel_id
	In []:  2**51 < to_legacy_ampel_id("ZTF32zzzzzz") < 2**52
	Out []: True
	"""

	if isinstance(ztf_id, str):
		num = 0
		s2 = ztf_id[5:]
		for i in rg:
			num += ab[s2[i]] * powers[i]
		return (num << 4) + enc_ztf_years[ztf_id[3:5]]

	return [to_ampel_id(name) for name in ztf_id]


@overload
def to_ztf_id(ampel_id: StockId) -> str:
	...

@overload
def to_ztf_id(ampel_id: StrictIterable[StockId]) -> List[str]:
	...

def to_ztf_id(ampel_id: Union[StockId, StrictIterable[StockId]]) -> Union[str, List[str]]:
	"""
	%timeit to_ztf_id(274878346346)
	1.54 µs ± 77.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
	"""
	# Handle sequences
	if isinstance(ampel_id, Iterable) and not isinstance(ampel_id, str):
		return [to_ztf_id(l) for l in ampel_id]
	elif isinstance(ampel_id, int):

		# int('00001111', 2) bitmask equals 15
		year = dec_ztf_years[ampel_id & 15]

		# Shift base10 encoded value 4 bits to the right
		ampel_id = ampel_id >> 4

		# Convert back to base26
		l = ['a', 'a', 'a', 'a', 'a', 'a', 'a']
		for i in rg:
			l[i] = alphabet[ampel_id % 26]
			ampel_id //= 26
			if not ampel_id:
				break

		return f"ZTF{year}{''.join(l)}"
	else:
		raise TypeError(f"Ampel ids for ZTF transients should be ints (got {type(ampel_id)} {ampel_id})")
