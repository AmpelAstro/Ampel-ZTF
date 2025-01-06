#!/usr/bin/env python3
# File:                Ampel-ZTF/ampel/ztf/t4/T4ZTFArchiveTokenGenerator.py
# License:             BSD-3-Clause
# Author:              valery brinnel & Simeon Reusch
# Date:                21.12.2022
# Last Modified Date:  07.04.2023
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

import random
import time
from datetime import datetime
from typing import Any

from astropy.time import Time
from requests_toolbelt.sessions import BaseUrlSession

from ampel.abstract.AbsT4Unit import AbsT4Unit
from ampel.secret.NamedSecret import NamedSecret
from ampel.struct.UnitResult import UnitResult
from ampel.types import UBson


class T4ZTFArchiveTokenGenerator(AbsT4Unit):
    archive_token: NamedSecret[str] = NamedSecret[str](label="ztf/archive/token")

    #: Base URL of archive service
    archive: str = "https://ampel.zeuthen.desy.de/api/ztf/archive/v3/"
    resource_name: str = "%%ztf_stream_token"

    max_dist_ps1_src: float = 0.5
    min_detections: int = 3

    date_str: None | str = None  # Start from a particular date
    date_format: str = "%Y-%m-%d"
    delta_t: float = 1.0  # Length of time window from start date (days)

    days_ago: None | float = None

    #: overrides max_dist_ps1_src & min_detections
    candidate: None | dict[str, Any] = None

    #: seconds to wait for query to complete
    timeout: float = 60

    debug: bool = False

    def do(self) -> UBson | UnitResult:
        if self.date_str:
            start_jd = Time(
                str(datetime.strptime(self.date_str, self.date_format)),  # noqa: DTZ007
                format="iso",
                scale="utc",
            ).jd
            end_jd = start_jd + self.delta_t

        elif self.days_ago:
            end_jd = Time.now().jd
            start_jd = end_jd - float(self.days_ago)

        else:
            start_jd = 2459899.04167
            end_jd = 2459899.045167

        if self.candidate:
            candidate = self.candidate
        else:
            candidate = {
                "distpsnr1": {"$lt": self.max_dist_ps1_src},
                "rb": {"$gt": 0.3},
                "magpsf": {"$lte": 20},
                "sgscore1": {"$lte": 0.9},
                "ndethist": {"$gte": self.min_detections},
                "isdiffpos": {"$in": ["t", "1"]},
                "nmtchps": {"$lte": 100},
            }

        session = BaseUrlSession(
            self.archive if self.archive.endswith("/") else self.archive + "/"
        )
        session.headers["authorization"] = f"bearer {self.archive_token.get()}"

        response = session.post(
            "streams/from_query",
            json={"jd": {"$gt": start_jd, "$lt": end_jd}, "candidate": candidate},
        )

        rd = response.json()
        try:
            token = rd.pop("resume_token")
        except KeyError as exc:
            raise ValueError(f"Unexpected response: {rd}") from exc

        # wait for query to finish
        t0 = time.time()
        delay = 1
        while time.time() - t0 < self.timeout:
            response = session.get(f"stream/{token}")
            if response.status_code not in (423, 404):
                break
            time.sleep(random.uniform(0, delay))
            delay *= 2
        else:
            raise RuntimeError(
                f"{session.base_url}stream/{token} still locked after {time.time() - t0:.0f} s"
            )
        response.raise_for_status()
        self.logger.info("Stream created", extra=response.json())

        return {self.resource_name: token}
