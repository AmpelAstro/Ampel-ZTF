#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/alert/ZiHealpixAlertSupplier.py
# License           : BSD-3-Clause
# Author            : Marcus Fenner <mf@physik.hu-berlin.de>
# Date              : 15.11.2021
# Last Modified Date: 14.06.2022
# Last Modified By  : Marcus Fenner <mf@physik.hu-berlin.de>

from datetime import datetime
from typing import Literal

from ampel.alert.BaseAlertSupplier import BaseAlertSupplier
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol
from ampel.ztf.alert.ZiAlertSupplier import ZiAlertSupplier
from ampel.ztf.alert.load.ZTFHealpixAlertLoader import HealpixSource

class ZiHealpixAlertSupplier(BaseAlertSupplier):
    """
    Iterable class that, for each alert payload provided by the underlying alert_loader,
    returns an AmpelAlertProtocol instance.
    """

    # Override default
    deserialize: None | Literal["avro", "json"] = None
    source: None | HealpixSource = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.source:
            self.alert_loader.source = self.source  # type: ignore[attr-defined]

    def set_healpix(
        self,
        nside: int,
        pixels: list[int],
        time: datetime,
        with_history: bool = False,
    ) -> None:
        """
        Define the Healpix map property which will be used by the loader.
        Nominally set in Loader config?
        """

        self.alert_loader.set_source(  # type: ignore[attr-defined]
            nside=nside, pixels=pixels, time=time
        )

    def __next__(self) -> AmpelAlertProtocol:
        """
        :returns: a dict with a structure that AlertConsumer understands
        :raises StopIteration: when alert_loader dries out.
        :raises AttributeError: if alert_loader was not set properly before this method is called
        """

        d = self._deserialize(next(self.alert_loader))  # type: ignore
        return ZiAlertSupplier.shape_alert_dict(d)
