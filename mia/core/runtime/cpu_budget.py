"""How many CPUs this process may actually use."""
from __future__ import annotations

import os
import socket

_CGROUP_V2 = "/sys/fs/cgroup/cpu.max"
_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"
_CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
_LSF_MCPU_HOSTS = "LSB_MCPU_HOSTS"
_LSF_HOSTS = "LSB_HOSTS"
_LSF_NUMPROC = "LSB_DJOB_NUMPROC"


def _own_cgroup_v2_path() -> str | None:
    try:
        lines = open(_PROC_SELF_CGROUP).read().splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3 or parts[0] != "0":
            continue
        sub = parts[2].strip()
        if sub in ("", "/"):
            return None
        return os.path.join(_CGROUP_ROOT, sub.lstrip("/"), "cpu.max")
    return None


def _norm_host(h: str) -> str:
    return h.split(".", 1)[0].lower()


def _short_hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return ""


def _slots_from_mcpu_hosts(value: str, me: str) -> int | None:
    toks = value.split()
    if not toks or len(toks) % 2:
        return None
    total = 0
    for host, n in zip(toks[0::2], toks[1::2]):
        if _norm_host(host) != me:
            continue
        try:
            k = int(n)
        except ValueError:
            return None
        if k <= 0:
            return None
        total += k
    return total or None


def _slots_from_lsb_hosts(value: str, me: str) -> int | None:
    n = sum(1 for h in value.split() if _norm_host(h) == me)
    return n or None


def _lsf_slots() -> int | None:
    me = _norm_host(_short_hostname())
    if me:
        for var, reader in ((_LSF_MCPU_HOSTS, _slots_from_mcpu_hosts),
                            (_LSF_HOSTS, _slots_from_lsb_hosts)):
            raw = os.environ.get(var)
            if raw:
                n = reader(raw, me)
                if n:
                    return n
    raw = os.environ.get(_LSF_NUMPROC)
    if raw:
        try:
            n = int(raw.strip())
        except ValueError:
            return None
        if n > 0:
            return n
    return None


def _from_v2(path: str) -> int | None:
    try:
        quota, period = open(path).read().split()[:2]
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    try:
        q, p = int(quota), int(period)
    except ValueError:
        return None
    if q <= 0 or p <= 0:
        return None
    return max(1, q // p)


def _from_v1() -> int | None:
    try:
        q = int(open(_CGROUP_V1_QUOTA).read().strip())
        p = int(open(_CGROUP_V1_PERIOD).read().strip())
    except (OSError, ValueError):
        return None
    if q <= 0 or p <= 0:
        return None
    return max(1, q // p)


def _kernel_cpus() -> int:
    for path in (_CGROUP_V2, _own_cgroup_v2_path()):
        if not path:
            continue
        n = _from_v2(path)
        if n:
            return n
    n = _from_v1()
    if n:
        return n
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def allocated_cpus() -> int:
    """CPUs this process may use; at least 1, never the machine count when a limit is visible."""
    n = _kernel_cpus()
    slots = _lsf_slots()
    if slots:
        return max(1, min(n, slots))
    return n
