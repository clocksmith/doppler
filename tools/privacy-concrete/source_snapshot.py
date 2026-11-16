"""Stable local public source materialization; metadata alone is never immutability."""
from contextlib import contextmanager
import hashlib
from pathlib import Path


def file_signature(path):
    value=Path(path).stat()
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


@contextmanager
def pinned_source(directory,files):
    """Bind whole-file identities and reject source changes across lazy client loading.

    This detects ordinary filesystem mutation; it is not protection from a
    malicious client host/kernel or a promise of a physically atomic filesystem.
    """
    directory=Path(directory);signatures={}
    def unchanged():
        if any(file_signature(directory/name)!=identity for name,identity in signatures.items()):
            raise ValueError('Pinned public source changed during materialization')
    for name,expected in files.items():
        path=directory/name;before=file_signature(path)
        with path.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
        if actual!=expected or file_signature(path)!=before:raise ValueError('Pinned public source required')
        signatures[name]=before
    unchanged()
    try:yield
    finally:unchanged()
