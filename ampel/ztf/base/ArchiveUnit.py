from functools import cached_property

import requests
from requests_toolbelt.sessions import BaseUrlSession

from ampel.core.ContextUnit import ContextUnit
from ampel.secret.NamedSecret import NamedSecret


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token: str) -> None:
        self.token = token

    def __call__(self, req: requests.PreparedRequest) -> requests.PreparedRequest:
        req.headers["authorization"] = f"bearer {self.token}"
        return req


class ArchiveUnit(ContextUnit):
    """
    Base class for interacting with the DESY ZTF alert archive
    """

    archive_token: NamedSecret[str] = NamedSecret(label="ztf/archive/token")

    # NB: init lazily, as Secret properties are not resolved until after __init__()
    @cached_property
    def session(self) -> BaseUrlSession:
        """Pre-authorized requests.Session"""
        session = BaseUrlSession(
            base_url=(
                url
                if (
                    url := self.context.config.get(
                        "resource.ampel-ztf/archive", str, raise_exc=True
                    )
                ).endswith("/")
                else url + "/"
            )
        )
        session.auth = BearerAuth(self.archive_token.get())
        return session
