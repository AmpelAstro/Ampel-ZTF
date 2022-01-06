#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/t1/ZiT1RetroCombiner.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                25.05.2021
# Last Modified Date:  25.05.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from typing import Generator

from ampel.content.DataPoint import DataPoint
from ampel.types import DataPointId
from ampel.t1.T1SimpleRetroCombiner import T1SimpleRetroCombiner
from ampel.ztf.t1.ZiT1Combiner import ZiT1Combiner

class ZiT1RetroCombiner(T1SimpleRetroCombiner, ZiT1Combiner): # type: ignore[misc]
	"""
	In []: zi_retro_combiner = ZiT1RetroCombiner(logger=AmpelLogger.get_logger(), access=[], policy=[])
	In []: [el.dps for el in zi_retro_combiner.combine([{'id': 7}, {'id': 6}, {'id': 5}])]
	Out[]: [[7, 6, 5], [6, 5], [5]]

	In []: [el.dps for el in zi_retro_combiner.combine([{'id': 7}, {'id': -6}, {'id': 5}])]
	Out[]: [[7, -6, 5], [5]]

	In []: [el.dps for el in zi_retro_combiner.combine([{'id': 7}, {'id': -6}, {'id': -5}])]
	Out[]: [[7, -6, -5]]

	"""

	def generate_retro_sequences(self, datapoints: list[DataPoint]) -> Generator[list[DataPointId], None, None]:
		datapoints = sorted(datapoints, key=lambda dp: dp["body"]["jd"])
		print([dp["tag"][-1] for dp in datapoints])
		while datapoints:
			yield [dp["id"] for dp in datapoints]
			# trim the list at the next most recent differential photometric point
			for i in range(len(datapoints)-2, -1, -1):
				if "ZTF_DP" in datapoints[i]["tag"]:
					datapoints = datapoints[:i+1]
					break
			else:
				break

