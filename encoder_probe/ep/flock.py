"""Exclusive whole-file locks that the OS releases if the holder dies: fcntl.flock on POSIX, msvcrt on Windows."""
from __future__ import annotations

import time

try:
    import fcntl
except ImportError:
    fcntl = None
    import msvcrt


def try_lock(f) -> bool:
    if fcntl is not None:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
    try:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)   # lock byte 0 (works on an empty file)
        return True
    except OSError:
        return False


def lock(f) -> None:
    if fcntl is not None:
        fcntl.flock(f, fcntl.LOCK_EX)
        return
    while not try_lock(f):
        time.sleep(1)


def unlock(f) -> None:
    if fcntl is not None:
        fcntl.flock(f, fcntl.LOCK_UN)
        return
    f.seek(0)
    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
