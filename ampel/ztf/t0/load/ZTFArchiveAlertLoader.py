import logging
from typing import Dict, Any, Union, Optional

import backoff
import requests

from ampel.base.AmpelBaseModel import AmpelBaseModel
from ampel.model.StrictModel import StrictModel

log = logging.getLogger(__name__)


class ObjectSource(StrictModel):
    #: A ZTF name
    ztf_name: str
    jd_start: Optional[float] = None
    jd_end: Optional[float] = None
    with_history: bool = True
    archive_token: str


class ZTFArchiveAlertLoader(AmpelBaseModel):
    #: Base URL of archive service
    archive: str = "https://ampel.zeuthen.desy.de/api/ztf/archive/v2"
    #: A stream identifier, created via POST /api/ztf/archive/streams/, or a query
    stream: Union[str, ObjectSource]

    def __iter__(self):
        return self.get_alerts()
    
    def get_alerts(self):
        with requests.Session() as session:
            while True:
                chunk = self._get_chunk(session)
                try:
                    yield from chunk["alerts"] if isinstance(chunk, dict) else chunk
                except GeneratorExit:
                    log.error(
                        f"Chunk from stream {self.stream} partially consumed. "
                        f"Ensure that iter_max is a multiple of the chunk size."
                    )
                    raise
                if (
                    isinstance(self.stream, ObjectSource)
                    or (len(chunk["alerts"]) == 0 and chunk["chunks_remaining"] == 0)
                ):
                    break

    @backoff.on_exception(
        backoff.expo,
        requests.HTTPError,
        giveup=lambda e: e.response.status_code not in {503, 504, 429, 408},
        max_time=600,
    )
    def _get_chunk(self, session: requests.Session) -> Dict[str, Any]:
        if isinstance(self.stream, ObjectSource):
            response = session.get(
                f"{self.archive}/object/{self.stream.ztf_name}/alerts",
                headers={"Authorization": f"bearer {self.stream.archive_token}"},
                params={
                    "with_history": self.stream.with_history,
                    **({"jd_start": self.stream.jd_start} if self.stream.jd_start is not None else {}), # type: ignore[dict-item]
                    **({"jd_end": self.stream.jd_end} if self.stream.jd_end is not None else {}), # type: ignore[dict-item]
                }
            )
        else:
            response = session.get(f"{self.archive}/stream/{self.stream}/chunk")
        response.raise_for_status()
        return response.json()