#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/alert/load/ZTFHealpixAlertLoader.py
# License           : BSD-3-Clause
# Author            : Marcus Fenner <mf@physik.hu-berlin.de>
# Date              : 9.11.2021
# Last Modified Date: 05.05.2022
# Last Modified By  : Marcus Fenner <mf@physik.hu-berlin.de>

from datetime import datetime
from functools import cached_property
from typing import (
    Any,
    Generator,
    Iterator,
)

import backoff
import requests
from ampel.abstract.AbsAlertLoader import AbsAlertLoader
from ampel.base.AmpelBaseModel import AmpelBaseModel
from ampel.log.AmpelLogger import AmpelLogger
from ampel.ztf.base.ArchiveUnit import BearerAuth, BaseUrlSession
from ampel.secret.NamedSecret import NamedSecret

from astropy.time import Time


class HealpixSource(AmpelBaseModel):
    #: Parameters for a Healpix query
    nside: int = 128
    pixels: list[int] = []
    time: datetime
    with_history: bool = False

class ZTFHealpixAlertLoader(AbsAlertLoader[dict[str, Any]]):
    """
    Create iterator of alerts found within a Healpix map.
    """

    history_days: float = 30.
    future_days: float = 30.
    chunk_size: int = 500
    query_size: int = 500  # number of ipix to query in one request
    query_start: int = 0  # first ipix index to query
    with_history: bool = False

    archive: str = "https://ampel.zeuthen.desy.de/api/ztf/archive/v3/"

    #: A stream identifier, created via POST /api/ztf/archive/streams/
    stream: None | str = None
    #: A HealpixSource object to query
    source: None | HealpixSource = None
    # If not set at init, needs to be set by alert proceessor

    archive_token: NamedSecret[str] = NamedSecret(label="ztf/archive/token")

    # NB: init lazily, as Secret properties are not resolved until after __init__()
    @cached_property
    def session(self) -> BaseUrlSession:
        """Pre-authorized requests.Session"""
        session = BaseUrlSession(
            base_url=(
                url
                if (
                    url := self.archive
                ).endswith("/")
                else url + "/"
            )
        )
        session.auth = BearerAuth(self.archive_token.get())
        return session
    
    class Config:
        """
        This is needed to not get pickle errors with python3.10
        see https://github.com/samuelcolvin/pydantic/issues/1241
        """
        keep_untouched = (cached_property,)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.logger: AmpelLogger = AmpelLogger.get_logger()
        self._it: None | Iterator[dict[str, Any]] = None

    def set_logger(self, logger: AmpelLogger) -> None:
        self.logger = logger

    def set_source(
        self,
        nside: int,
        pixels: list[int],
        time: datetime,
        with_history: bool = False,
    ) -> None:
        self.source = HealpixSource(
            nside=nside,
            pixels=pixels,
            time=time,
            with_history=with_history,
        )
        # Reset iter
        self._it = None

    def __iter__(self) -> Iterator[dict[str, Any]]:  # type: ignore[override]
        return self._get_alerts()

    def __next__(self) -> dict[str, Any]:
        if self._it is None:
            self._it = iter(self)
        return next(self._it)

    def _get_alerts(self) -> Generator[dict[str, Any], None, None]:
        assert self.source is not None
        while self.query_start < len(self.source.pixels):  # type: ignore[union-attr]
            chunk = self._get_chunk()
            if self.stream is None:
                self.stream = chunk["resume_token"]
            try:
                yield from chunk["alerts"] if isinstance(
                    chunk, dict
                ) else chunk
            except GeneratorExit:
                self.logger.error(
                    f"Chunk from stream {self.stream} partially consumed."
                )
                raise GeneratorExit
            if chunk["remaining"]["chunks"] == 0:
                self.query_start += self.query_size
                self.stream = None

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.HTTPError,
        giveup=lambda e: not isinstance(e, requests.HTTPError) or e.response.status_code not in {500, 502, 503, 504, 429, 408},
        max_time=600,
    )
    def _get_chunk(self) -> dict[str, Any]:
        if self.stream is None:
            jd = Time(self.source.time, scale="utc").jd  # type: ignore[union-attr]
            response = self.session.post(
                "alerts/healpix/skymap",
                json={
                    "nside": self.source.nside,  # type: ignore[union-attr]
                    "pixels": self.source.pixels,  # type: ignore[union-attr]
                    "with_history": str(self.with_history),
                    "with_cutouts": "false",
                    "jd": {
                        "$gt": jd - self.history_days,
                        "$lt": jd + self.future_days,
                    },
                    "chunk_size": self.chunk_size,
                    "latest": "false"
                }
            )
        else:
            response = self.session.get(
                f"{self.archive}/stream/{self.stream}/chunk"
            )
        response.raise_for_status()
        return response.json()
