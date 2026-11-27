"""Conservative per-worker reference-geometry budget, without new dependencies."""
import os
import sys
from pathlib import Path

MIB = 1024 * 1024


def _system_available_memory_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except (ImportError, OSError, AttributeError, NotImplementedError):
        pass
    try:
        if sys.platform == 'win32':
            import ctypes
            class Status(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total',
                     'page_available', 'virtual_total', 'virtual_available', 'extended')]
            status = Status(); status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available)
        elif sys.platform.startswith('linux'):
            from pathlib import Path
            for line in Path('/proc/meminfo').read_text().splitlines():
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
        # macOS without psutil: retain the modest fixed budget rather than
        # interpreting total or free pages as memory available to this job.
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _linux_cgroup_headroom(proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    """Account for readable v1/v2 limits, including tighter ancestor groups."""
    bases=[Path(cgroup_root),Path(cgroup_root)/'memory']
    candidates={bases[0]}
    if (bases[1]/'memory.limit_in_bytes').is_file():candidates.add(bases[1])
    try:
        for line in (Path(proc_root)/'self/cgroup').read_text().splitlines():
            _,controllers,group=line.split(':',2)
            if controllers and 'memory' not in controllers.split(','):continue
            base=bases[1] if controllers else bases[0]
            current=base/group.lstrip('/')
            if '..' in current.parts:continue
            while current==base or base in current.parents:
                candidates.add(current)
                if current==base:break
                current=current.parent
    except (OSError,ValueError):pass
    headroom=[]
    for directory in candidates:
        for limit_name,usage_name in (('memory.max','memory.current'),
                                      ('memory.limit_in_bytes','memory.usage_in_bytes')):
            try:
                limit=int((directory/limit_name).read_text().strip())
                used=int((directory/usage_name).read_text().strip())
                if limit>=0 and used>=0:headroom.append(max(0,limit-used))
            except (OSError,ValueError):pass  # Includes the v2 unlimited 'max'.
    return min(headroom) if headroom else None


def available_memory_bytes():
    available=_system_available_memory_bytes()
    if sys.platform.startswith('linux'):
        limited=_linux_cgroup_headroom()
        if limited is not None:
            available=limited if available is None else min(available,limited)
    return available


def reference_budget(workers, available):
    """Use at most a quarter of available RAM across all requested workers.

    Budgets estimate retained mesh/BVH data, not RSS. Leave substantial room
    for path arrays, in-flight queries, temporary allocations and other apps.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('Height worker count must be a positive integer.')
    if available is None:
        return 256 * MIB
    per_worker = max(1,int(available) // (4 * workers))
    # A one-byte budget effectively disables retention under known exhaustion;
    # do not turn a low/zero reading into the unknown-memory 256 MiB fallback.
    rounded=(per_worker//MIB)*MIB if per_worker>=MIB else per_worker
    return min(4096*MIB,rounded)
