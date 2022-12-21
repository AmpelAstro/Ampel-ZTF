#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/t3/resource/T3ZTFArchiveTokenGenerator.py
# License:             BSD-3-Clause
# Author:              valery brinnel & Simeon Reusch
# Date:                21.12.2022
# Last Modified Date:  21.12.2022
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

import requests
from typing import Any
from astropy.time import Time  # type: ignore
from datetime import datetime

from ampel.types import UBson
from ampel.struct.T3Store import T3Store
from ampel.struct.Resource import Resource
from ampel.struct.UnitResult import UnitResult
from ampel.secret.NamedSecret import NamedSecret
from ampel.abstract.AbsT3PlainUnit import AbsT3PlainUnit


class T3ZTFArchiveTokenGenerator(AbsT3PlainUnit):

	archive_token: NamedSecret[str] = NamedSecret(label="ztf/archive/token")
	archive_stream_endpoint: str = "https://ampel.zeuthen.desy.de/api/ztf/archive/v3/streams/from_query?"
	resource_name: str = 'ztf_stream_token'

	max_dist_ps1_src: float = 0.5
	min_detections: int = 3

	date_str: None | str = None
	date_format: str = "%Y-%m-%d"
	days_ago: None | int = None

	#: overrides max_dist_ps1_src & min_detections
	candidate: None | dict[str, Any] = None

	debug: bool = False


	def process(self, t3s: T3Store) -> UBson | UnitResult:

		if self.date_str:
			start_jd = Time(
				str(datetime.strptime(self.date_str, self.date_format)),
				format="iso", scale="utc"
			).jd
			delta_t = 1
			end_jd = start_jd + delta_t

		elif self.days_ago:
			end_jd = Time.now().jd
			start_jd = end_jd - float(self.days_ago)

		else:
			start_jd = 2459899.04167
			end_jd = 2459899.045167


		if self.candidate:
			candidate = self.candidate
		else:
			candidate = {
				"distpsnr1": {"$lt": self.max_dist_ps1_src},
				"rb": {"$gt": 0.3},
				"magpsf": {"$lte": 20},
				"sgscore1": {"$lte": 0.9},
				"ndethist": {"$gte": self.min_detections},
				"isdiffpos": {"$in": ["t", "1"]},
				"nmtchps": {"$lte": 100},
			}

		response = requests.post(
			self.archive_stream_endpoint,
			headers = {"Authorization": f"bearer {self.archive_token}"},
			json = {
				"jd": {"$gt": start_jd, "$lt": end_jd},
				"candidate": candidate
			}
		)

		rd = response.json()
		if "resume_token" not in rd:
			raise ValueError(f"Unexpected response: {rd}")

		r = Resource(name=self.resource_name, value=rd["resume_token"])
		t3s.add_resource(r)

		if self.debug:
			return r.dict()

		return None
