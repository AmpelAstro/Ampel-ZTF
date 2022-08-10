#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/ztf/t0/load/ZTFAlertArchiver.py
# License:             BSD-3-Clause
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                14.04.2021
# Last Modified Date:  14.04.2021
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>


import io
import time
from typing import Any

import fastavro

from ampel.abstract.AbsOpsUnit import AbsOpsUnit
from ampel.secret.NamedSecret import NamedSecret
from ampel.ztf.t0.load.AllConsumingConsumer import AllConsumingConsumer

try:
    from ampel.ztf.t0.ArchiveUpdater import ArchiveUpdater
except ImportError:
    ...


class ZTFAlertArchiver(AbsOpsUnit):

    #: Address of Kafka broker
    bootstrap: str = "partnership.alerts.ztf.uw.edu:9092"
    #: Consumer group name
    group_name: str
    #: Topic name regexes to subscribe to
    topics: list[str] = ["^ztf_.*_programid1$", "^ztf_.*_programid2$"]
    #: Time to wait for messages before giving up, in seconds
    timeout: int = 300
    #: URI of postgres server hosting the archive
    archive_uri: str
    archive_auth: NamedSecret[dict] = NamedSecret(label="ztf/archive/writer")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.archive_updater = ArchiveUpdater(
            self.archive_uri,
            connect_args=self.archive_auth.get(),
        )

        self.consumer = AllConsumingConsumer(
            self.bootstrap,
            timeout=self.timeout,
            topics=self.topics,
            **{"group.id": self.group_name},
        )

    def run(self, beacon: None | dict[str, Any] = None) -> None | dict[str, Any]:

        try:
            for message in self.consumer:
                reader = fastavro.reader(io.BytesIO(message.value()))
                alert = next(reader)  # raise StopIteration
                self.archive_updater.insert_alert(
                    alert,
                    reader.writer_schema,
                    message.partition(),
                    int(1e6 * time.time()),
                )
        except KeyboardInterrupt:
            ...
        
        return None
