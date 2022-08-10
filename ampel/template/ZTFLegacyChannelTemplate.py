#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/template/ZTFLegacyChannelTemplate.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                16.10.2019
# Last Modified Date:  30.05.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from typing import Any, ClassVar
from ampel.log.AmpelLogger import AmpelLogger
from ampel.config.builder.FirstPassConfig import FirstPassConfig
from ampel.template.AbsEasyChannelTemplate import AbsEasyChannelTemplate
from ampel.model.ingest.T2Compute import T2Compute


class ZTFLegacyChannelTemplate(AbsEasyChannelTemplate):
	"""
	Channel template for ZTF. Each of the named variants consumes adifferent
	alert streams from IPAC, and produce stocks with a different set of tags:

	============== ============== ========================
	Template       ZTF programids Tags
	============== ============== ========================
	ztf_uw_private 1, 2, 3_public ZTF, ZTF_PUB, ZTF_PRIV
	ztf_uw_public  1, 3_public    ZTF, ZTF_PUB
	============== ============== ========================
	"""

	# static variables (ClassVar type) are ignored by pydantic
	_access: ClassVar[dict[str, list[str]]] = {
		"ztf_uw_private": ["ZTF", "ZTF_PUB", "ZTF_PRIV"],
		"ztf_uw_public": ["ZTF", "ZTF_PUB"],
		"ztf_uw_caltech": ["ZTF", "ZTF_PUB"],
	}

	auto_complete: Any = False

	#: include all previously ingested photopoints in emitted states
	live_history: bool = True
	#: include X days of archival datapoints in emitted states
	archive_history: None | int = None

	# Mandatory implementation
	def get_channel(self, logger: AmpelLogger) -> dict[str, Any]:
		assert self.template is not None
		return {
			**super().get_channel(logger),
			"access": self.__class__._access[self.template],
		}

	# Mandatory implementation
	def get_processes(
		self, logger: AmpelLogger, first_pass_config: FirstPassConfig
	) -> list[dict[str, Any]]:

		# T3 processes
		ret: list[dict[str, Any]] = []

		for index, el in enumerate(self.t3_supervise):
			# populate name and tier if unset
			name = el.get("name", f"summary_{index:02d}")
			process_name = f"{self.channel}|T3|{name}"
			ret.append(
				self.transfer_channel_parameters(el | {"name": process_name, "tier": 3})
			)

		if not any(model.unit == "T2LightCurveSummary" for model in self.t2_compute):
			self.t2_compute.append(T2Compute(unit="T2LightCurveSummary"))

		mongo_muxer = {"unit": "ZiMongoMuxer"} if self.live_history else None
		archive_muxer = (
			{"unit": "ZiArchiveMuxer", "config": {"history_days": self.archive_history}}
			if self.archive_history is not None
			else None
		)
		if mongo_muxer and archive_muxer:
			muxer: None | dict[str,Any] = {
				"unit": "ChainedT0Muxer",
				"config": {"muxers": [mongo_muxer, archive_muxer]},
			}
		elif mongo_muxer:
			muxer = mongo_muxer
		elif archive_muxer:
			muxer = archive_muxer
		else:
			muxer = None

		supplier = {
			"unit": "ZiAlertSupplier",
			"config": {
				"loader": {
					"unit": "UWAlertLoader",
					"config": {
						**first_pass_config["resource"]["ampel-ztf/kafka"],
						**{"stream": self.template},
					},
				}
			}
		}

		ret.insert(
			0,
			self.craft_t0_process(
				first_pass_config,
				controller="ZTFAlertStreamController",
				supplier=supplier,
				shaper="ZiDataPointShaper",
				muxer=muxer,
				combiner={"unit": "ZiT1Combiner", "config": {"access": self.access, "policy": self.policy}},
			),
		)



		return ret
