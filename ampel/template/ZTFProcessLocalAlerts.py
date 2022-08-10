#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/template/ZTFProcessLocalAlerts.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                16.07.2021
# Last Modified Date:  24.11.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from typing import Any, Literal
from ampel.types import ChannelId
from ampel.log.AmpelLogger import AmpelLogger
from ampel.model.UnitModel import UnitModel
from ampel.model.ingest.T2Compute import T2Compute
from ampel.abstract.AbsProcessorTemplate import AbsProcessorTemplate
from ampel.template.AbsEasyChannelTemplate import AbsEasyChannelTemplate


class ZTFProcessLocalAlerts(AbsProcessorTemplate):
	"""
	Returns adequate config for an alert consumer configured to process local alerts
	"""

	channel: ChannelId
	folder: str
	extension: Literal['json', 'avro', 'csv'] = "json"

	#: Note: if a UnitModel is provided as supplier config entries of keys
	#: 'deserialize' and 'loader' will be overriden
	supplier: str | UnitModel = 'ZiAlertSupplier'
	loader: str = 'DirAlertLoader'
	binary_mode: None | bool = True

 	#: T2 units to trigger when transient is updated. Dependencies of tied
	#: units will be added automatically.
	t2_compute: list[T2Compute] = []

	extra: dict = {}


	# Mandatory override
	def get_model(self, config: dict[str, Any], logger: AmpelLogger) -> UnitModel:

		return UnitModel(
			unit = 'AlertConsumer',
			config = self.extra | AbsEasyChannelTemplate.craft_t0_processor_config(
				channel = self.channel,
				config = config,
				t2_compute = self.t2_compute,
				supplier = self._get_supplier(),
				shaper = "ZiDataPointShaper",
				combiner = "ZiT1Combiner",
				filter_dict = None,
				muxer = None,
				compiler_opts = {
					'stock': {'id_mapper': 'ZTFIdMapper', 'tag': 'ZTF'},
					't0': {'tag': 'ZTF'},
					't1': {'tag': 'ZTF'},
					'state_t2': {'tag': 'ZTF'},
					'point_t2': {'tag': 'ZTF'}
				}
			)
		)

	def _get_supplier(self) -> dict[str, Any]:

		d: dict[str, Any] = {
			'unit': self.supplier if isinstance(self.supplier, str) else self.supplier.unit,
			'config': {
				'deserialize': self.extension,
				'loader': {
					'unit': self.loader,
					'config': self._get_loader_conf()
				}
			}
		}

		if isinstance(self.supplier, UnitModel):
			d['config'] = self.supplier.config | d['config']

		return d


	def _get_loader_conf(self) -> dict[str, Any]:

		d: dict[str, Any] = {
			'folder': self.folder,
			'extension': self.extension
		}

		if self.binary_mode is not None:
			d['binary_mode'] = self.binary_mode

		return d
