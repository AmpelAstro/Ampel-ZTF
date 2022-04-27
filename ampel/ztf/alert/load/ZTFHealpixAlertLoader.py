#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/alert/load/ZTFHealpixAlertLoader.py
# License           : BSD-3-Clause
# Author            : mf <mf@physik.hu-berlin.de>
# Date              : 9.11.2021
# Last Modified Date: 27.04.2022
# Last Modified By  : jno <jnordin@physik.hu-berlin.de>

from datetime import datetime
from functools import cached_property
from json import JSONDecodeError
from typing import Any, Dict, List, Generic, Iterator, Union, Optional

import backoff
import numpy as np
import requests
from ampel.types import T

from ampel.abstract.AbsAlertLoader import AbsAlertLoader
from ampel.base.AmpelBaseModel import AmpelBaseModel
from ampel.log.AmpelLogger import AmpelLogger
from ampel.secret.NamedSecret import NamedSecret
from ampel.model.StrictModel import StrictModel
from astropy.time import Time
import healpy as hp
from requests_toolbelt.sessions import BaseUrlSession
from ampel.ztf.base.ArchiveUnit import BearerAuth

class HealpixSource(StrictModel):
    #: Parameters for a Healpix query
    nside: int
    pixels: List[int]
    time: datetime
    with_history: str = 'false'


class ZTFHealpixAlertLoader(AbsAlertLoader):
    """
    Create iterator of alerts found within a Healpix map.
    """

    history_days: int = 30
    future_days: int = 30
    chunk_size: int = 500
    # Do not know how to set this through a job file
    archive_token: Union[str, NamedSecret[str]] = NamedSecret(label="ztf/archive/token")

    archive: str = "https://ampel.zeuthen.desy.de/api/ztf/archive/v3/"
    #: V2. A stream identifier, created via POST /api/ztf/archive/streams/, or a query
    # If not set at init, needs to be set by alert proceessor
    stream: Optional[Union[str, HealpixSource]]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.logger: AmpelLogger = AmpelLogger.get_logger()
        self._it : Optional[Iterator] = None

    def set_logger(self, logger: AmpelLogger) -> None:
        self.logger = logger

    def set_source(self, nside: int, pixels: List[int], time: datetime, with_history: bool = False) -> None:
        self.stream = HealpixSource(
            nside=nside,
            pixels=pixels,
            time=time,
            with_history=with_history,
        )
        # Reset iter
        self._it = None

    @cached_property
    def session(self) -> BaseUrlSession:
        session = BaseUrlSession(base_url=(self.archive))
        if isinstance(self.archive_token, NamedSecret):
            session.auth = BearerAuth(self.archive_token.get())
        else:
            session.auth = self.archive_token
        return session


    def __iter__(self):
        return self.get_alerts()

    def __next__(self) -> T:
        if not self._it:
            self._it = iter(self)
        return next(self._it)


    def get_alerts(self):
        with requests.Session() as session:
            while True:
                chunk = self._get_chunk(session)
                try:
                    yield from chunk["alerts"] if isinstance(chunk, dict) else chunk
                except GeneratorExit:
                    self.logger.error(
                        f"Chunk from stream {self.stream} partially consumed. "
                        f"Ensure that iter_max is a multiple of the chunk size."
                    )
                    raise
                # Simple reimplementation thaat I understand...
                if (
                    isinstance(self.stream, HealpixSource)
                    or (len(chunk["alerts"]) == 0 and chunk["chunks_remaining"] == 0)
                ):
                    break
                if isinstance(self.stream, HealpixSource):
                    self.stream = chunk['resume_token']


    @backoff.on_exception(
        backoff.expo,
        requests.HTTPError,
        giveup=lambda e: e.response.status_code not in {500, 502, 503, 504, 429, 408},
        max_time=600,
    )
    def _get_chunk(self, session: requests.Session) -> Dict[str, Any]:
        if isinstance(self.stream, HealpixSource):
            jd = Time(self.stream.time, scale="utc").jd
            response = session.post(
                f"{self.archive}alerts/healpix/skymap",
                headers={"Authorization": f"bearer {self.archive_token}"},
                json = {
                    "nside": self.stream.nside,
                    "pixels": self.stream.pixels,
                    "with_history": self.stream.with_history,
                    "with_cutouts": "false",
                    "jd": {
                        "$gt": jd - self.history_days,
                        "$lt": jd + self.future_days,
                    },
                    "chunk_size": self.chunk_size,
                }
            )
        else:
            response = session.get(f"{self.archive}/stream/{self.stream}/chunk")
        response.raise_for_status()
        return response.json()
