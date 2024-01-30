import grp
import io
import json
import os
import pwd
import tarfile
import time
from functools import lru_cache
from pathlib import Path

import fastavro


@lru_cache
def schema(version):
    base = Path(__file__).parents[2] / "test" / "test-data"
    with open(base / f"schema_{version}.json") as f:
        return json.load(f)


def dump(alert, fileobj):
    fastavro.writer(fileobj, schema(alert["schemavsn"]), [alert])


def to_tarball(alert_generator, **tarfile_kwargs):
    """
    Write alert dicts to a tar archive
    """
    uid = pwd.getpwuid(os.geteuid()).pw_name
    gid = grp.getgrgid(os.getegid()).gr_name
    euid = os.geteuid()
    egid = os.getegid()
    with tarfile.open(mode="w:gz", **tarfile_kwargs) as archive:
        i = 0
        total_bytes = 0
        for i, alert in enumerate(alert_generator):  # noqa: B007
            with io.BytesIO() as payload:
                dump(alert, payload)
                ti = tarfile.TarInfo(f"{alert['candid']}.avro")
                ti.size = len(payload.getvalue())
                total_bytes += ti.size
                ti.mtime = time.time()
                ti.uid = euid
                ti.uname = uid
                ti.gid = egid
                ti.gname = gid
                payload.seek(0)
                archive.addfile(ti, payload)
        return {"alerts": i + 1, "total_bytes": total_bytes}
