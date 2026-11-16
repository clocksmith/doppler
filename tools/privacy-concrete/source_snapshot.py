"""Stable local public source materialization; metadata alone is never immutability."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from protocol import require_active


def file_signature(path):
    value=Path(path).stat()
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


@contextmanager
def pinned_source(directory,files,*,cancelled=None):
    """Bind whole-file identities and reject source changes across lazy client loading.

    This detects ordinary filesystem mutation; it is not protection from a
    malicious client host/kernel or a promise of a physically atomic filesystem.
    """
    require_active(cancelled)
    directory=Path(directory);signatures={}
    def unchanged():
        if any(file_signature(directory/name)!=identity for name,identity in signatures.items()):
            raise ValueError('Pinned public source changed during materialization')
    for name,expected in files.items():
        require_active(cancelled)
        path=directory/name;before=file_signature(path)
        hasher=hashlib.sha256()
        with path.open('rb') as handle:
            while True:
                require_active(cancelled)
                chunk=handle.read(1024*1024)
                require_active(cancelled)
                if not chunk:break
                hasher.update(chunk)
        actual=hasher.hexdigest()
        if actual!=expected or file_signature(path)!=before:raise ValueError('Pinned public source required')
        signatures[name]=before
    unchanged()
    try:
        require_active(cancelled)
        yield
        require_active(cancelled)
    finally:unchanged()
