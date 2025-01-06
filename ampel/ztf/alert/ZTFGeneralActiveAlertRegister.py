#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/alert/ZTFGeneralActiveAlertRegister.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                26.05.2020
# Last Modified Date:  27.06.2022
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from collections.abc import Sequence
from struct import pack
from typing import Any, BinaryIO, ClassVar

from ampel.log.AmpelLogger import AmpelLogger
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.ztf.alert.ZTFGeneralAlertRegister import ZTFGeneralAlertRegister


class ZTFGeneralActiveAlertRegister(ZTFGeneralAlertRegister):
    """
    Optimized GeneralAlertRegister in that ZTF stock saved with 5 bytes instead of 8 (std Q size).
    That is because:
    In []: 2**36 < to_ampel_id('ZTF33zzzzzzz') < 2**37
    Out[]: True
    Logs: alert_id, filter_res, stock
    """

    __slots__: ClassVar[tuple[str, ...]] = (
        "_write",
        "alert_max",
        "alert_min",
        "stock_max",
        "stock_min",
    )
    _slot_defaults = {
        "alert_max": 0,
        "alert_min": 2**64,
        "stock_max": 0,
        "stock_min": 2**64,
        "ztf_years": set(),
    }

    new_header_size: int | str = "+4096"
    header_hints: ClassVar[Sequence[str]] = ("alert", "stock")
    alert_min: int
    alert_max: int
    stock_min: int
    stock_max: int
    ztf_years: set[int]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        hdr = self.header["payload"]
        if "ztf_years" in hdr:
            self.ztf_years = set(hdr["ztf_years"])

    def file(self, alert: AmpelAlertProtocol, filter_res: int = 0) -> None:
        alid = alert.id
        self.alert_max = max(alid, self.alert_max)
        self.alert_min = min(alid, self.alert_min)

        sid = alert.stock
        self.stock_max = max(sid, self.stock_max)  # type: ignore[assignment]
        self.stock_min = min(sid, self.stock_min)  # type: ignore[assignment]

        if (sid & 15) not in self.ztf_years:  # type: ignore[operator]
            self.ztf_years.add(sid & 15)  # type: ignore[operator]

        self._write(pack("<QBQ", alert.id, -filter_res, alert.stock)[:-3])

    def close(self, **kwargs) -> None:  # type: ignore[override]
        hdr = self.header["payload"]
        if self.ztf_years:
            hdr["ztf_years"] = list(self.ztf_years)

        super().close(**kwargs)

    @classmethod
    def find_stock(  # type: ignore[override]
        cls,
        f: BinaryIO | str,
        stock_id: int | list[int],
        **kwargs,
    ) -> None | list[tuple[int, ...]]:
        return super().find_stock(
            f, stock_id, header_hint_callback=cls._match_ztf_years, **kwargs
        )

    @staticmethod
    def _match_ztf_years(
        header: dict[str, Any],
        stock_id: int | list[int],
        logger: None | AmpelLogger = None,
    ) -> None | int | list[int]:
        if "ztf_years" in header:
            zy = header["ztf_years"]
        else:
            return stock_id

        if isinstance(stock_id, int):
            if stock_id & 15 in zy:
                if logger:
                    logger.info(f"Header ZTF year check: {stock_id} is eligible")
                return stock_id

            if logger:
                logger.info(f"Header ZTF year check: {stock_id} is not eligible")
            return None

        ret = [el for el in stock_id if (el & 15) in zy]
        if len(ret) == len(stock_id):
            if logger:
                logger.info("Header ZTF year check: all stock IDs are eligible")
            return stock_id

        if logger:
            if len(ret) == 0:
                logger.info(
                    "Header ZTF year check: none of the provided stock IDs are eligible"
                )
            else:
                logger.info(
                    f"Header ZTF year check: stock IDs search targets reduced to {ret}"
                )

        return ret
