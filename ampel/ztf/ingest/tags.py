#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/ingest/tags.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                14.03.2020
# Last Modified Date:  18.03.2020
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from typing import Dict

# tags is used by ZiT0PhotoPointShaper and ZiT0UpperLimitShaper
# First key: programid, second key: filter id
tags: dict[int, dict[int, list[str]]] = {
	1: {
		1: ["ZTF", "ZTF_PUB", "ZTF_G"],
		2: ["ZTF", "ZTF_PUB", "ZTF_R"],
		3: ["ZTF", "ZTF_PUB", "ZTF_I"]
	},
	2: {
		1: ["ZTF", "ZTF_PRIV", "ZTF_G"],
		2: ["ZTF", "ZTF_PRIV", "ZTF_R"],
		3: ["ZTF", "ZTF_PRIV", "ZTF_I"]
	},
	3: { # Actually CALTEC
		1: ["ZTF", "ZTF_PUB", "ZTF_PRIV", "ZTF_G"],
		2: ["ZTF", "ZTF_PUB", "ZTF_PRIV", "ZTF_R"],
		3: ["ZTF", "ZTF_PUB", "ZTF_PRIV", "ZTF_I"]
	},
	0: {
		1: ["ZTF", "ZTF_PUB", "ZTF_PRIV", "ZTF_G"],
		2: ["ZTF", "ZTF_PUB", "ZTF_PRIV", "ZTF_R"],
		3: ["ZTF", "ZTF_PUB", "ZTF_PRIV", "ZTF_I"]
	}
}
