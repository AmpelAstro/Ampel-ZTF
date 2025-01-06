#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/util/ZTFIdMapper.py
# License:             BSD-3-Clause
# Author:              Simeon Reusch <simeon.reusch@desy.de
# Date:                19.02.2023
# Last Modified Date:  01.03.2023
# Last Modified By:    Simeon Reusch <simeon.reusch@desy.de

from collections.abc import Iterable
from typing import cast, overload

from ampel.abstract.AbsIdMapper import AbsIdMapper
from ampel.types import StockId, StrictIterable
from ampel.ztf.util.ZTFIdMapper import ZTFIdMapper


class ZTFNoisifiedIdMapper(AbsIdMapper):
    def __init__(self):
        ZTFIdMapper.__init__(self)

    @overload
    @classmethod
    def to_ampel_id(cls, ztf_id: str) -> int: ...

    @overload
    @classmethod
    def to_ampel_id(cls, ztf_id: StrictIterable[str]) -> list[int]: ...

    @classmethod
    def to_ampel_id(cls, ztf_id: str | StrictIterable[str]) -> int | list[int]:
        """
        Append an integer to a padded Ampel ID.
        This is useful for e.g. noisified versions of the
        same parent lightcurve
        """
        if isinstance(ztf_id, str):
            split_str = ztf_id.split("_")

            ampel_id = ZTFIdMapper.to_ampel_id(ztf_id)

            if len(split_str) > 1:
                sub_id = split_str[1]
                return int(str(ampel_id) + "000000" + sub_id)
            return ampel_id

        return [cast(int, cls.to_ampel_id(name)) for name in ztf_id]

    @overload
    @classmethod
    def to_ext_id(cls, ampel_id: StockId) -> str: ...

    @overload
    @classmethod
    def to_ext_id(cls, ampel_id: StrictIterable[StockId]) -> list[str]: ...

    @classmethod
    def to_ext_id(cls, ampel_id: StockId | StrictIterable[StockId]) -> str | list[str]:
        """
        Return the original name of the noisified lightcurve
        """
        if isinstance(ampel_id, Iterable) and not isinstance(ampel_id, str):
            return [cast(str, cls.to_ext_id(l)) for l in ampel_id]

        if isinstance(ampel_id, int):
            both_ids = str(ampel_id).split("000000")
            ampel_id = int(both_ids[0])

            ztfid = ZTFIdMapper.to_ext_id(ampel_id)

            if len(both_ids) > 1:
                sub_id = int(both_ids[1])
                return ztfid + "_" + str(sub_id)

            return ztfid

        raise TypeError(
            f"Ampel ids for ZTF transients should be ints (got {type(ampel_id)} {ampel_id})"
        )


# backward compatibility shortcuts
to_ampel_id = ZTFNoisifiedIdMapper.to_ampel_id
to_ztf_id = ZTFNoisifiedIdMapper.to_ext_id
