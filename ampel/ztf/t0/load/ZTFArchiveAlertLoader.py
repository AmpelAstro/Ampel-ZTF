#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/t0/load/ZTFArchiveAlertLoader.py
# License:             BSD-3-Clause
# Author:              jvs
# Date:                20.10.2021
# Last Modified Date:  22.12.2022
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

import logging, backoff, requests
from typing import Any
from ampel.abstract.AbsAlertLoader import AbsAlertLoader
from ampel.base.AmpelBaseModel import AmpelBaseModel
from ampel.struct.Resource import Resource

log = logging.getLogger(__name__)


class ZTFSource(AmpelBaseModel):
    ztf_name: str
    jd_start: None | float = None
    jd_end: None | float = None
    with_history: bool = True
    archive_token: str


class ZTFArchiveAlertLoader(AbsAlertLoader):
    """
    Load ZTF alerts from a stream provided by DESY alert archive.
    The stream is initiated either by an archive stream token
    (directly via loader config parameter or via method add_resource)
    or an archive query formulated as ZTFSource.
    get_alerts yields chunks of alerts until consumed, at which point
    this is acknowledged and a new chunk retrieved.
    """

    #: Base URL of archive service
    archive: str = "https://ampel.zeuthen.desy.de/api/ztf/archive/v3"

    #: A stream identifier, created via POST /api/ztf/archive/streams/, or a query
    stream: None | str | ZTFSource

    #: Name of dynamic resource, fetched by a T3 process and forwarded
    #: to suppliers/loaders by AlertConsumer via their methods add_resource
    resource_name: str = 'ztf_stream_token'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._it = None

    # override
    def add_resource(self, name: str, resource: Resource) -> None:
        if name == self.resource_name:
            if not isinstance(resource.value, str):
                raise ValueError(
                    f"Unexpected {self.resource_name} resource "
                    f"value type: {type(resource.value)}"
                )
            self.stream = resource.value

    def __iter__(self):
        return self.get_alerts()

    def __next__(self):
        if not self._it:
            self._it = iter(self)
        return next(self._it)

    def get_alerts(self):
        if self.stream is None:
            raise ValueError()
        with requests.Session() as session:
            while True:
                chunk = self._get_chunk(session)
                yield from chunk["alerts"] if isinstance(chunk, dict) else chunk
                # NB: if generator exits before we get here, chunk is never acknowledged
                if "chunk" in chunk:
                    self._acknowledge_chunk(session, chunk["chunk"])
                    log.info(
                        None, extra={"streamToken": self.stream, "chunk": chunk["chunk"]}
                    )
                if isinstance(self.stream, ZTFSource) or (
                    len(chunk["alerts"]) == 0 and chunk["remaining"]["chunks"] == 0
                ):
                    break

    @backoff.on_exception(
        backoff.expo,
        requests.HTTPError,
        giveup = lambda e: (
            not isinstance(e, requests.HTTPError) or
            e.response.status_code not in {502, 503, 504, 429, 408}
        ),
        max_time = 600,
    )
    def _get_chunk(self, session: requests.Session) -> dict[str, Any]:

        if isinstance(self.stream, ZTFSource):
            params = {"with_history": self.stream.with_history}
            for k in ("jd_start", "jd_end"):
                if (x := getattr(self.stream, k)) is not None:
                    params[k] = x
            response = session.get(
                f"{self.archive}/object/{self.stream.ztf_name}/alerts",
                headers = {"Authorization": f"bearer {self.stream.archive_token}"},
                params = params
            )
        else:
            response = session.get(f"{self.archive}/stream/{self.stream}/chunk")
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(
        backoff.expo,
        requests.HTTPError,
        giveup = (
            lambda e: not isinstance(e, requests.HTTPError) or
            e.response.status_code not in {502, 503, 504, 429, 408}
        ),
        max_time = 600,
    )
    def _acknowledge_chunk(self, session: requests.Session, chunk_id: int) -> None:
        response = session.post(f"{self.archive}/stream/{self.stream}/chunk/{chunk_id}/acknowledge")
        response.raise_for_status()
