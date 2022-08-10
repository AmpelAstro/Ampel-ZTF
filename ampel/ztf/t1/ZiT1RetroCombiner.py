#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/t1/ZiT1RetroCombiner.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                25.05.2021
# Last Modified Date:  25.05.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from ampel.t1.T1PhotoRetroCombiner import T1PhotoRetroCombiner
from ampel.ztf.t1.ZiT1Combiner import ZiT1Combiner

class ZiT1RetroCombiner(T1PhotoRetroCombiner, ZiT1Combiner): # type: ignore[misc]
	"""
	In []: zi_retro_combiner = ZiT1RetroCombiner(logger=AmpelLogger.get_logger(), access=[], policy=[])
	In []: [el.dps for el in zi_retro_combiner.combine([{'id': 7}, {'id': 6}, {'id': 5}])]
	Out[]: [[7, 6, 5], [6, 5], [5]]

	In []: [el.dps for el in zi_retro_combiner.combine([{'id': 7}, {'id': -6}, {'id': 5}])]
	Out[]: [[7, -6, 5], [5]]

	In []: [el.dps for el in zi_retro_combiner.combine([{'id': 7}, {'id': -6}, {'id': -5}])]
	Out[]: [[7, -6, -5]]

	Note: class is empty, it's fine
	"""
