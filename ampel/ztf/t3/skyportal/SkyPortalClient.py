#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/t3/skyportal/SkyPortalClient.py
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                16.09.2020
# Last Modified Date:  24.11.2022
# Last Modified By:    Simeon Reusch <simeon.reusch@desy.de>

import base64
import gzip
import io
import json
import math
import time
from collections import defaultdict
from collections.abc import Generator, Iterable, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypedDict, overload
from urllib.parse import urlparse

import backoff
import numpy as np
import requests
from astropy.io import fits
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from ampel.base.AmpelBaseModel import AmpelBaseModel
from ampel.base.AmpelUnit import AmpelUnit
from ampel.enum.DocumentCode import DocumentCode
from ampel.log.AmpelLogger import AmpelLogger
from ampel.metrics.AmpelMetricsRegistry import AmpelMetricsRegistry
from ampel.protocol.LoggerProtocol import LoggerProtocol
from ampel.secret.NamedSecret import NamedSecret
from ampel.types import Traceless
from ampel.util.collections import ampel_iter
from ampel.util.mappings import flatten_dict

if TYPE_CHECKING:
    from ampel.config.AmpelConfig import AmpelConfig
    from ampel.content.DataPoint import DataPoint
    from ampel.view.T2DocView import T2DocView
    from ampel.view.TransientView import TransientView


stat_http_errors = AmpelMetricsRegistry.counter(
    "http_request_errors",
    "HTTP request failures",
    subsystem=None,
    labelnames=("method", "endpoint"),
)
stat_http_responses = AmpelMetricsRegistry.counter(
    "http_responses",
    "HTTP responses successfully recieved",
    subsystem=None,
    labelnames=("method", "endpoint"),
)
stat_http_time = AmpelMetricsRegistry.histogram(
    "http_request_time",
    "Duration of HTTP requests",
    unit="seconds",
    subsystem=None,
    labelnames=("method", "endpoint"),
)
stat_concurrent_requests = AmpelMetricsRegistry.gauge(
    "http_requests_inprogress",
    "Number of HTTP requests in flight",
    subsystem=None,
    labelnames=("method", "endpoint"),
    multiprocess_mode="livesum",
)


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, tuple | list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def encode_t2_body(t2: "T2DocView") -> str:
    assert t2.body is not None
    doc = t2.body[-1]
    assert isinstance(doc, dict)
    return base64.b64encode(
        json.dumps(
            {
                "timestamp": datetime.fromtimestamp(
                    t2.meta[-1]["ts"], tz=timezone.utc
                ).isoformat(),
                **{k: sanitize_json(v) for k, v in doc.items() if k != "ts"},
            },
            default=lambda o: None,
        ).encode()
    ).decode()


def decode_t2_body(blob: str | dict[str, Any]) -> dict[str, Any]:
    doc = (
        json.loads(base64.b64decode(blob.encode()).decode())
        if isinstance(blob, str)
        else blob
    )
    return {"ts": int(datetime.fromisoformat(doc.pop("timestamp")).timestamp()), **doc}


def get_t2_result(
    t2: "T2DocView",
) -> tuple[None, None] | tuple[datetime, dict[str, Any]]:
    assert t2.body is not None
    for meta, record in zip(reversed(t2.meta), reversed(t2.body), strict=False):  # noqa: B007
        if meta.get("code", DocumentCode.OK) == DocumentCode.OK:
            break
    else:
        return None, None
    assert isinstance(record, dict)
    return datetime.fromtimestamp(meta["ts"], tz=timezone.utc), record


def render_thumbnail(cutout_data: bytes) -> str:
    """
    Render gzipped FITS as base64-encoded PNG
    """
    with gzip.open(io.BytesIO(cutout_data), "rb") as f, fits.open(f) as hdu:
        img = np.flipud(hdu[0].data)
    mask = np.isfinite(img)

    fig = Figure(figsize=(1, 1))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.imshow(
        img,
        # clip pixel values below the median
        norm=Normalize(*np.percentile(img[mask], [0.5, 99.5])),
        aspect="auto",
        origin="lower",
    )

    with io.BytesIO() as buf:
        fig.savefig(buf, dpi=img.shape[0])
        return base64.b64encode(buf.getvalue()).decode()


ZTF_FILTERS = {1: "ztfg", 2: "ztfr", 3: "ztfi"}


class CutoutSpec(AmpelBaseModel):
    #: where to find cutouts
    key: str = "ZTFCutoutImages"
    #: mapping from cutout names to SkyPortal thumbnail types
    types: dict[str, str] = {
        "cutoutScience": "new",
        "cutoutTemplate": "ref",
        "cutoutDifference": "sub",
    }


class SkyPortalAPIError(IOError): ...


class SkyPortalClient(AmpelUnit):
    #: Base URL of SkyPortal server
    base_url: str
    #: API token
    token: NamedSecret[str]
    #: Maximum number of in-flight requests
    max_parallel_connections: int = 1

    @classmethod
    def validate(cls, value: dict) -> Any:
        value = super().validate(value)
        url = urlparse(value["base_url"])
        if url.scheme not in ("http", "https"):
            raise ValueError("base_url must be http(s)")
        if value["base_url"].endswith("/"):
            raise ValueError("base_url may not have a path set")
        return value

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._ids: dict[str, dict[str, int]] = {}
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"token {self.token.get()}"

    @overload
    def request(
        self,
        verb: str,
        endpoint: str,
        *,
        raise_exc: bool = True,
        headers: None | dict[str, str] = None,
        params: None | dict[str, Any] = None,
        json: None | dict[str, Any] = None,
        _decode_json: None,
    ) -> requests.Response: ...

    @overload
    def request(
        self,
        verb: str,
        endpoint: str,
        *,
        raise_exc: bool = True,
        headers: None | dict[str, str] = None,
        params: None | dict[str, Any] = None,
        json: None | dict[str, Any] = None,
        _decode_json: bool = True,
    ) -> dict[str, Any]: ...

    @backoff.on_exception(
        backoff.expo,
        (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
        max_time=1200,
        factor=10,
    )
    def request(
        self,
        verb: str,
        endpoint: str,
        *,
        raise_exc: bool = True,
        headers: None | dict[str, str] = None,
        params: None | dict[str, Any] = None,
        json: None | dict[str, Any] = None,
        _decode_json: None | bool = True,
    ) -> requests.Response | dict[str, Any]:
        if endpoint.startswith("/"):
            url = self.base_url + endpoint
        else:
            url = self.base_url + "/api/" + endpoint
        labels = (verb, endpoint.split("/")[0])
        with (
            stat_http_time.labels(*labels).time(),
            stat_http_errors.labels(*labels).count_exceptions(
                (
                    requests.exceptions.HTTPError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                )
            ),
            stat_concurrent_requests.labels(*labels).track_inprogress(),
        ):
            response = self._session.request(
                verb,
                url,
                headers=headers,
                params=params,
                json=json,
            )
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            stat_http_responses.labels(*labels).inc()
            if _decode_json:
                payload = response.json()
                if raise_exc:
                    # only check status if endpoint knows it was returning JSON
                    if (
                        response.headers["content-type"] == "application/json"
                        and payload["status"] != "success"
                    ):
                        raise SkyPortalAPIError(payload["message"], url)
                    # otherwise, believe status code
                    response.raise_for_status()
                return payload
            return response

    def get_id(
        self,
        endpoint: str,
        params: dict[str, Any],
        default: None | dict[str, Any] = None,
    ) -> int:
        """Query for an object by id, inserting it if not found"""
        if not (response := self.get(endpoint, params=params, raise_exc=False))["data"]:
            response = self.post(endpoint, json=default or params)
        if isinstance(response["data"], list):
            return response["data"][0]["id"]
        return response["data"]["id"]

    def get_by_name(self, endpoint: str, name: str) -> int:
        if endpoint not in self._ids:
            self._ids[endpoint] = {}
        if name not in self._ids[endpoint]:
            self._ids[endpoint][name] = self._get_by_name(endpoint, name)
        return self._ids[endpoint][name]

    def _get_by_name(self, endpoint: str, name: str) -> int:
        try:
            return next(
                d["id"]
                for d in (self.get(endpoint, params={"name": name}))["data"]
                if d["name"] == name
            )
        except StopIteration:
            pass
        raise KeyError(f"No {endpoint} named {name}")

    def get(
        self,
        endpoint: str,
        *,
        params: None | dict[str, Any] = None,
        raise_exc: bool = True,
    ) -> dict[str, Any]:
        return self.request("GET", endpoint, params=params, raise_exc=raise_exc)

    def post(
        self,
        endpoint: str,
        *,
        params: None | dict[str, Any] = None,
        json: None | dict[str, Any] = None,
        raise_exc: bool = True,
    ) -> dict[str, Any]:
        return self.request(
            "POST", endpoint, params=params, json=json, raise_exc=raise_exc
        )

    def put(
        self,
        endpoint: str,
        *,
        params: None | dict[str, Any] = None,
        json: None | dict[str, Any] = None,
        raise_exc: bool = True,
    ) -> dict[str, Any]:
        return self.request(
            "PUT", endpoint, params=params, json=json, raise_exc=raise_exc
        )

    def head(self, endpoint: str) -> requests.Response:
        return self.request("HEAD", endpoint, _decode_json=None)


class FilterGroupProvisioner(SkyPortalClient):
    """
    Set up filters to corresponding to AMPEL channels
    """

    #: mapping from ampel stream name to Fritz stream name
    stream_names: dict[str, str] = {
        "ztf_uw_public": "ZTF Public",
        "ztf_uw_private": "ZTF Public+Partnership",
        "ztf_uw_caltech": "ZTF Public+Partnership+Caltech",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_filter(self, name, stream, group):
        try:
            return self.get_by_name("filters", name)
        except KeyError:
            ...
        doc = {
            "name": name,
            "stream_id": self.get_by_name("streams", self.stream_names[stream]),
            "group_id": self.get_by_name("groups", group),
        }
        self.post("filters", json=doc)
        return self.get_by_name("filters", name)

    def create_filters(
        self, config: "AmpelConfig", group: str, stream: None | str = None
    ) -> None:
        """
        Create a dummy SkyPortal filter for each Ampel filter

        :param group: name of group that should own the filter
        :param stream:
          name of the filter's alert stream (meaningless, since there is no
          filter actually defined)
        """
        for channel in config.get("channel", dict, raise_exc=True).values():
            if not channel.get("active", True):
                continue
            name = f"AMPEL.{channel['channel']}"
            try:
                self.get_by_name("filters", name)
                continue
            except KeyError:
                ...
            self.create_filter(name, stream or channel["template"], group)


def provision_seed_data(client: SkyPortalClient):
    """Set up instruments and groups for a test instance"""
    p48 = client.get_id(
        "telescope",
        {"name": "P48"},
        {
            "diameter": 1.2,
            "elevation": 1870.0,
            "lat": 33.3633675,
            "lon": -116.8361345,
            "nickname": "Palomar 1.2m Oschin",
            "name": "P48",
            "skycam_link": "http://bianca.palomar.caltech.edu/images/allsky/AllSkyCurrentImage.JPG",
            "robotic": True,
        },
    )

    source = {
        "instrument": client.get_id(
            "instrument",
            {"name": "ZTF"},
            {
                "filters": ["ztfg", "ztfr", "ztfi"],
                "type": "imager",
                "band": "optical",
                "telescope_id": p48,
                "name": "ZTF",
            },
        ),
        "stream": client.get_id("streams", {"name": "ztf_partnership"}),
        "group": 1,  # root group
    }
    if source["stream"] not in [
        groupstream["id"]
        for groupstream in (client.get(f"groups/{source['group']}"))["data"]["streams"]
    ]:
        client.post(
            f"groups/{source['group']}/streams", json={"stream_id": source["stream"]}
        )
    source["filter"] = client.get_id(
        "filters",
        {"name": "highlander"},
        {
            "name": "highlander",
            "stream_id": source["stream"],
            "group_id": source["group"],
        },
    )

    # ensure that all users are in the root group
    users = [
        users["username"]
        for users in (client.get(f"groups/{source['group']}"))["data"]["users"]
    ]
    for user in (client.get("user"))["data"]:
        if user["username"] not in users:
            client.post(
                f"groups/{source['group']}/users",
                json={"username": user["username"]},
            )

    return source


class PostReport(TypedDict):
    new: bool  #: is this a new source?
    candidates: list[int]  #: new candidates created
    save_error: None | str  #: error raised while saving source
    photometry_count: int  #: size of posted photometry
    photometry_error: None | str  #: error raised while posting photometry
    thumbnail_count: int  #: number of thumbnails posted
    comments: int  #: number of comments posted
    comment_errors: list[str]  #: errors raised while posting comments
    dt: float  #: elapsed time in seconds


class BaseSkyPortalPublisher(SkyPortalClient):
    logger: Traceless[LoggerProtocol]

    def __init__(self, **kwargs):
        if "logger" not in kwargs:
            kwargs["logger"] = AmpelLogger.get_logger()
        super().__init__(**kwargs)

    def _transform_datapoints(
        self, dps: Sequence["DataPoint"], after=-float("inf")
    ) -> Generator[dict[str, Any], None, None]:
        for dp in dps:
            body = dp["body"]
            if body["jd"] <= after:
                continue
            if "SUPERSEDED" in dp["tag"]:
                continue
            base = {
                "id": dp["id"],
                "filter": ZTF_FILTERS[body["fid"]],
                "mjd": body["jd"] - 2400000.5,
                "limiting_mag": body["diffmaglim"],
            }
            if body.get("magpsf") is not None:
                content = {
                    "mag": body["magpsf"],
                    "magerr": body["sigmapsf"],
                    "ra": body["ra"],
                    "dec": body["dec"],
                }
            else:
                content = {k: None for k in ("mag", "magerr", "ra", "dec")}
            yield {**base, **content}

    def make_photometry(
        self, datapoints: Sequence["DataPoint"], after=-float("inf")
    ) -> dict[str, Any]:
        content = defaultdict(list)
        for doc in self._transform_datapoints(datapoints, after):
            for k, v in doc.items():
                content[k].append(v)
        return dict(content)

    def _find_instrument(self, tags: Sequence[int | str]) -> int:
        with suppress(KeyError, requests.exceptions.RequestException):
            for tag in tags:
                if isinstance(tag, str):
                    return self.get_by_name("instrument", tag)
        raise KeyError(f"None of {tags} match a known instrument")

    def post_t2_annotations(
        self,
        name: str,
        t2_views: Iterable["T2DocView"],
        object_record: None | dict[str, Any],
        ret: PostReport,
    ):
        previous_annotations = object_record["annotations"] if object_record else []
        for t2 in t2_views:
            last_modified, result = get_t2_result(t2)
            if result is None or last_modified is None:
                continue
            # find associated annotation
            for annotation in previous_annotations:
                if ":".split(annotation["origin"])[-1] == t2.unit:
                    break
            else:
                # post new annotation
                self.logger.debug(f"posting {t2.unit}")
                try:
                    self.post(
                        f"sources/{name}/annotation",
                        json={
                            "origin": f"ampel:{t2.unit}",
                            "data": flatten_dict(result, ":"),
                        },
                    )
                    ret["comments"] += 1
                except SkyPortalAPIError as exc:
                    ret["comment_errors"].append(exc.args[0])
                continue
            # update previous annotation
            if last_modified > datetime.fromisoformat(annotation["updated"]):
                self.logger.debug(f"updating {t2.unit}")
                try:
                    self.put(
                        f"sources/{name}/annotation/{annotation['id']}",
                        json={
                            "obj_id": name,
                            "author_id": annotation["author_id"],
                            "origin": f"ampel:{t2.unit}",
                            "data": flatten_dict(result, ":"),
                        },
                    )
                    ret["comments"] += 1
                except SkyPortalAPIError as exc:
                    ret["comment_errors"].append(exc.args[0])
            else:
                self.logger.debug(f"{t2.unit} exists and is current")

    def post_t2_comments(
        self,
        name: str,
        t2_views: Iterable["T2DocView"],
        object_record: None | dict[str, Any],
        ret: PostReport,
    ):
        previous_comments = object_record["comments"] if object_record else []

        for t2 in t2_views:
            # find associated comment
            for comment in previous_comments:
                if comment["text"] == t2.unit:
                    break
            else:
                # post new comment
                self.logger.debug(f"posting {t2.unit}")
                try:
                    self.post(
                        f"sources/{name}/comment",
                        json={
                            "text": t2.unit,
                            "attachment": {
                                "body": encode_t2_body(t2),
                                "name": f"{name}-{t2.unit}.json",
                            },
                        },
                    )
                    ret["comments"] += 1
                except SkyPortalAPIError as exc:
                    ret["comment_errors"].append(exc.args[0])
                continue
            # update previous comment
            previous_body = decode_t2_body(
                self.get(
                    f"sources/{name}/comments/{comment['id']}/attachment",
                )
            )
            if (t2.body is not None) and (t2.meta[-1]["ts"] > previous_body["ts"]):
                self.logger.debug(f"updating {t2.unit}")
                try:
                    self.put(
                        f"sources/{name}/comment/{comment['id']}",
                        json={
                            "attachment_bytes": encode_t2_body(t2),
                            "author_id": comment["author_id"],
                            "obj_id": name,
                            "text": comment["text"],
                        },
                    )
                    ret["comments"] += 1
                except SkyPortalAPIError as exc:
                    ret["comment_errors"].append(exc.args[0])
            else:
                self.logger.debug(f"{t2.unit} exists and is current")

    def get_filter_ids(
        self, view: "TransientView", filters: Sequence[str] | None = None
    ) -> dict[str, int]:
        assert view.stock, f"{self.__class__} requires stock records"
        return {
            name: self.get_by_name("filters", name)
            for name in (
                filters
                or [f"AMPEL.{channel}" for channel in ampel_iter(view.stock["channel"])]
            )
        }

    def get_filter_updates(
        self,
        view: "TransientView",
        passed_filters: Sequence[int],
        filters: None | Sequence[str] = None,
    ) -> Generator[tuple[int, datetime, list[int]], None, None]:
        """
        Generates updates for filters based on journal entries in a given view.
        Args:
            view (TransientView): The view containing journal entries and stock information.
            passed_filters (Sequence[int]): A sequence of filter IDs that have already been passed.
            filters (None | Sequence[str], optional): A sequence of filter names to check. If None, defaults to filters based on the channels in the view's stock.
        Yields:
            tuple[int, datetime, list[int]]: A tuple containing the alert ID, the timestamp of the journal entry, and a list of filter IDs that passed for the first time.
        """

        assert view.stock, f"{self.__class__} requires stock records"
        new_filters = {
            self.get_by_name("filters", name)
            for name in (
                filters
                or [f"AMPEL.{channel}" for channel in ampel_iter(view.stock["channel"])]
            )
        }.difference(passed_filters)
        # Walk through the non-autocomplete journal entries in time order,
        # finding the time and alert id at which each filter passed for the
        # first time.
        for jentry in view.get_journal_entries(tier=0):
            if jentry.get("extra", {}).get("ac", False):
                continue
            # no more filters left to update
            if not new_filters:
                break
            # set all filters to passing on the first go if filters were
            # specified explicitly
            fids = (
                list(new_filters)
                if filters
                else [
                    fid
                    for channel in ampel_iter(jentry["channel"])
                    if (fid := self.get_by_name("filters", f"AMPEL.{channel}"))
                    in new_filters
                ]
            )
            # no new filters passed on this alert
            if not fids:
                continue
            yield (
                jentry["alert"],  # type: ignore[typeddict-item]
                datetime.fromtimestamp(jentry["ts"], tz=timezone.utc),
                fids,
            )
            new_filters.difference_update(fids)

    def post_candidate(
        self,
        view: "TransientView",
        *,
        filters: None | list[str] = None,
    ):
        """Post candidate for this object. post_source() must be called first."""
        name = self.get_source_name(view)

        if (
            response := self.get(
                f"candidates/{name}",
                params={"includeAlerts": 1},
                raise_exc=False,
            )
        )["status"] == "success":
            # Only update filters, not the candidate itself
            passed_filters = response["data"]["filter_ids"]
        else:
            passed_filters = []

        # Walk through the non-autocomplete journal entries in time order,
        # finding the time and alert id at which each filter passed for the
        # first time.
        for alert_id, passed_at, fids in self.get_filter_updates(
            view, passed_filters, filters
        ):
            self.post(
                "candidates",
                json={
                    "id": name,
                    "filter_ids": fids,
                    "passing_alert_id": alert_id,
                    "passed_at": passed_at.isoformat(),
                },
            )

    def get_source_name(self, view: "TransientView") -> str:
        assert view.stock, f"{self.__class__} requires stock records"
        assert (
            "name" in view.stock
        ), f"{self.__class__} requires stocks with a `name` field. Did you remember to set AlertConsumer.compiler_opts?"
        assert view.stock["name"] is not None
        return next(
            n for n in view.stock["name"] if isinstance(n, str) and n.startswith("ZTF")
        )

    def post_source(
        self,
        view: "TransientView",
        *,
        groups: list[str],  # groups are required
        annotate: bool = False,
    ) -> PostReport:
        """
        Perform the following actions:
          * Post candidate to filters specified by ``filters``. ``ra``/``dec``
            are taken from the most recent detection; ``drb`` from the maximum
            over detections.
          * Post photometry using the ``candid`` of the latest detection.
          * Post a PNG-encoded cutout of the last detection image.
          * Post each T2 result as comment with a JSON-encoded attachment. If
            a comment corresponding to the T2 unit already exists, overwrite it
            with the most recent result.

        :param view:
            Data to post
        :param filters:
            Names of the filter to associate with the candidate. If None, use
            filters named AMPEL.{channel} for each ``channel`` the transient
            belongs to.
        :param groups:
            Names of the groups to save the source to. If None, use all
            accessible groups associated with token.
        :param instrumentname:
            Name of the instrument with which to associate the photometry.
        """

        ret: PostReport = {
            "new": True,
            "candidates": [],
            "save_error": None,
            "photometry_count": 0,
            "photometry_error": None,
            "thumbnail_count": 0,
            "comments": 0,
            "comment_errors": [],
            "dt": -time.time(),
        }

        name = self.get_source_name(view)

        # Check if source exists. If not, instruct Kowalski to create it
        group_ids = {self.get_by_name("groups", name) for name in groups}
        if self.head(f"sources/{name}").status_code == 404:
            self.post(f"alerts/{name}", json={"group_ids": list(group_ids)})

        # represent latest T2 results as a comments
        latest_t2: dict[str, T2DocView] = {}
        for t2 in view.t2 or []:
            if t2.code != DocumentCode.OK or not t2.body:
                continue
            assert isinstance(t2.unit, str)
            if (
                t2.unit not in latest_t2
                or latest_t2[t2.unit].meta[-1]["ts"] < t2.meta[-1]["ts"]
            ):
                latest_t2[t2.unit] = t2

        # Get the source record (now guaranteed to exist)
        source = self.get(
            f"sources/{name}",
            params={
                "includeComments": 1 if latest_t2 and not annotate else 0,
            },
        )["data"]
        if groups_to_post := group_ids.difference(
            group["id"] for group in source["groups"]
        ):
            try:
                self.post(
                    "sources", json={"id": name, "group_ids": list(groups_to_post)}
                )
            except SkyPortalAPIError as exc:
                ret["save_error"] = exc.args[0]

        if latest_t2:
            if annotate:
                self.post_t2_annotations(
                    name,
                    latest_t2.values(),
                    source,
                    ret,
                )
            else:
                self.post_t2_comments(
                    name,
                    latest_t2.values(),
                    source,
                    ret,
                )

        ret["dt"] += time.time()

        return ret
