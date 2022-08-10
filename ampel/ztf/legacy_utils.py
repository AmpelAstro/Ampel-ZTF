#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/legacy_utils.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                07.06.2018
# Last Modified Date:  19.03.2020
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from typing import overload, Tuple


number_map = {
	'10': 'a', '11': 'b', '12': 'c', '13': 'd', '14': 'e', '15': 'f',
	'16': 'g', '17': 'h', '18': 'i', '19': 'j', '20': 'k', '21': 'l',
	'22': 'm', '23': 'n', '24': 'o', '25': 'p', '26': 'q', '27': 'r',
	'28': 's', '29': 't', '30': 'u', '31': 'v', '32': 'w', '33': 'x',
	'34': 'y', '35': 'z'
}

letter_map = {
	'a': '10', 'b': '11', 'c': '12', 'd': '13', 'e': '14', 'f': '15',
	'g': '16', 'h': '17', 'i': '18', 'j': '19', 'k': '20', 'l': '21',
	'm': '22', 'n': '23', 'o': '24', 'p': '25', 'q': '26', 'r': '27',
	's': '28', 't': '29', 'u': '30', 'v': '31', 'w': '32', 'x': '33',
	'y': '34', 'z': '35'
}


@overload
def to_ampel_id(ztf_id: str) -> int:
	...

@overload
def to_ampel_id(ztf_id: list[str] | tuple[str, ...]) -> list[int]:
	...

def to_ampel_id(ztf_id: str | list[str] | tuple[str, ...]) -> int | list[int]:
	""" :returns: Ampel ID (int).  """

	# Handle sequences
	if isinstance(ztf_id, str):
		return int(
			"".join(
				(
					ztf_id[3:5],
					letter_map[ztf_id[5]],
					letter_map[ztf_id[6]],
					letter_map[ztf_id[7]],
					letter_map[ztf_id[8]],
					letter_map[ztf_id[9]],
					letter_map[ztf_id[10]],
					letter_map[ztf_id[11]]
				)
			)
		)

	return [to_ampel_id(name) for name in ztf_id]

@overload
def to_ztf_id(ampel_id: int) -> str:
	...

@overload
def to_ztf_id(ampel_id: list[int] | tuple[int, ...]) -> list[str]:
	...

def to_ztf_id(ampel_id: int | list[int] | tuple[int, ...]) -> str | list[str]:
	"""
	:param ampel_id: int or list of int
	:returns: ZTF ID (string).
	"""
	# Handle sequences
	if isinstance(ampel_id, (list, tuple)):
		return [to_ztf_id(l) for l in ampel_id]

	str_long = str(ampel_id)

	return "ZTF%s%s%s%s%s%s%s%s" % (
		str_long[0:2],
		number_map[str_long[2:4]],
		number_map[str_long[4:6]],
		number_map[str_long[6:8]],
		number_map[str_long[8:10]],
		number_map[str_long[10:12]],
		number_map[str_long[12:14]],
		number_map[str_long[14:16]]
	)
