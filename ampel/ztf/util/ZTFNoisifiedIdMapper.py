#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/util/ZTFIdMapper.py
# License:             BSD-3-Clause
# Author:              Simeon Reusch <simeon.reusch@desy.de
# Date:                19.02.2023
# Last Modified Date:  01.03.2023
# Last Modified By:    Simeon Reusch <simeon.reusch@desy.de

from ampel.abstract.AbsIdMapper import AbsIdMapper
from ampel.types import StockId
from ampel.ztf.util.ZTFIdMapper import ZTFIdMapper


class ZTFNoisifiedIdMapper(AbsIdMapper):
    def __init__(self):
        ZTFIdMapper.__init__(self)

    @classmethod
    def to_ampel_id(cls, ztf_id: str) -> int:
        """
        Append an integer to a padded Ampel ID.
        This is useful for e.g. noisified versions of the
        same parent lightcurve
        """
        split_str = ztf_id.split("_")
        ampel_id_part = split_str[0]

        ampel_id = ZTFIdMapper.to_ampel_id(ztf_id)

        if len(split_str) > 1:
            sub_id = split_str[1]
            return int(str(ampel_id) + "000000" + sub_id)

        else:
            return ampel_id

    @classmethod
    def to_ext_id(cls, ampel_id_with_sub_id: StockId) -> str:
        """
        Return the original name of the noisified lightcurve
        """
        both_ids = str(ampel_id_with_sub_id).split("000000")
        ampel_id = int(both_ids[0])

        ztfid = ZTFIdMapper.to_ext_id(ampel_id)

        if len(both_ids) > 1:
            sub_id = int(both_ids[1])
            return ztfid + "_" + str(sub_id)

        else:
            return ztfid

# backward compatibility shortcuts
to_ampel_id = ZTFNoisifiedIdMapper.to_ampel_id
to_ztf_id = ZTFNoisifiedIdMapper.to_ext_id
