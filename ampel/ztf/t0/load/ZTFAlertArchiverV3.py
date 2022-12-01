#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : ampel/ztf/t0/load/ZTFAlertArchiverV3.py
# License           : BSD-3-Clause
# Author            : Jakob van Santen <jakob.van.santen@desy.de>
# Date              : 20.01.2022
# Last Modified Date: 20.01.2021
# Last Modified By  : Jakob van Santen <jakob.van.santen@desy.de>


import io
from functools import cached_property
from typing import Any, Optional, Iterator

import fastavro

from ampel.abstract.AbsOpsUnit import AbsOpsUnit
from ampel.ztf.base.ArchiveUnit import ArchiveUnit
from ampel.ztf.t0.load.AllConsumingConsumer import AllConsumingConsumer


class ZTFAlertArchiverV3(AbsOpsUnit, ArchiveUnit):
    """
    Ingest chunks of alerts into the ZTF alert archive using the v3 API
    """

    #: Address of Kafka broker
    bootstrap: str = "partnership.alerts.ztf.uw.edu:9092"
    #: Consumer group name
    group_name: str
    #: Topic name regexes to subscribe to
    topics: list[str] = ["^ztf_.*_programid1$", "^ztf_.*_programid2$"]
    #: Time to wait for messages before giving up, in seconds
    timeout: int = 300
    #: Number of alerts to post at once
    chunk_size: int = 1000
    #: extra configuration to pass to confluent_kafka.Consumer
    kafka_consumer_properties: dict[str,Any] = {}

    @cached_property
    def consumer(self) -> AllConsumingConsumer:
        return AllConsumingConsumer(
            self.bootstrap,
            timeout=self.timeout,
            topics=self.topics,
            auto_commit=False,
            logger=self.logger,
            **{"group.id": self.group_name},
            **self.kafka_consumer_properties,
        )

    def _chunks(self) -> Iterator[bytes]:
        """
        Yield avro-serialized chunks of alerts from consumer, i.e. strip the schema header
        from all but the first alert.
        """
        schema = None
        alerts: list = []

        # wrap in a generator so that offsets are not committed until
        # the _next_ iteration of the loop. this ensures that the offsets
        # are only committed once the chunk is actually archived.
        def emit() -> Iterator[bytes]:
            nonlocal schema
            if not alerts:
                return
            chunk = io.BytesIO()
            fastavro.writer(chunk, schema, alerts)
            yield chunk.getvalue()
            alerts.clear()
            schema = None
            self.consumer.commit()

        for message in self.consumer:
            reader = fastavro.reader(io.BytesIO(message.value()))
            alert = next(reader)  # raise StopIteration
            if schema is None:
                schema = reader.writer_schema
            alerts.append(alert)
            if len(alerts) >= self.chunk_size:
                yield from emit()
        yield from emit()

    def _post_chunk(self, payload: bytes):
        self.logger.debug(f"Posting chunk of size {len(payload)}")
        response = self.session.post("alerts", data=payload)
        response.raise_for_status()

    def run(self, beacon: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:

        try:
            for chunk in self._chunks():
                self._post_chunk(chunk)
        except KeyboardInterrupt:
            ...

        return None
