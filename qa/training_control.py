"""Cross-process exclusive training lock, compatible with Windows and Linux."""
from contextlib import contextmanager
from pathlib import Path
import os

@contextmanager
def training_lock(path='logs/training.lock'):
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    stream=target.open('a+b'); stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError('Another training process already holds the project lock') from None
    try: yield
    finally:
        stream.seek(0)
        if os.name=='nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()
