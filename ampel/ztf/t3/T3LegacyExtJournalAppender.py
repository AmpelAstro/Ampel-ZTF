#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/t3/T3LegacyExtJournalAppender.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                17.06.2020
# Last Modified Date:  14.08.2020
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from ampel.types import StockId
from ampel.content.JournalRecord import JournalRecord
from ampel.t3.supply.complement.T3ExtJournalAppender import T3ExtJournalAppender
from ampel.ztf.legacy_utils import to_ampel_id as legacy_to_ampel_id
from ampel.ztf.util.ZTFIdMapper import to_ztf_id


class T3LegacyExtJournalAppender(T3ExtJournalAppender):
	""" Allows to import journal entries from a v0.6.x ampel DB """


	def get_ext_journal(self, stock_id: StockId) -> None | list[JournalRecord]:
		"""
		Particularities:
		- converts stock id into the old encoding to perform DB search
		- rename field 'dt' into 'ts' to allow sorting by timestamp
		"""

		if ext_stock := next(
			self.col.find(
				{'_id': legacy_to_ampel_id(to_ztf_id(stock_id))} # type: ignore[arg-type]
			), None
		):
			for j in ext_stock['journal']:
				j['ts'] = j.pop('dt')
			if self.journal_filter:
				return self.journal_filter.apply(ext_stock['journal'])
			return ext_stock['journal']

		return None
