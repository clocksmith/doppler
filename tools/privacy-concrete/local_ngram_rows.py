"""Installed PLE table contract over shared verified client-local BF16 row storage."""
import json
from pathlib import Path
from local_bf16_rows import LocalBF16Rows,LocalRowsLease


class LocalNgramRows(LocalBF16Rows):
    def __init__(self,directory,inventory,*,local_storage_limit_bytes):
        super().__init__(directory,inventory,local_storage_limit_bytes=local_storage_limit_bytes,
                         layout_schema='doppler.flash-local-ngram-storage-inventory/v1')

    @classmethod
    def flash(cls,directory,*,local_storage_limit_bytes):
        inventory=json.loads(Path(__file__).with_name('flash-ngram-storage.json').read_text())
        return cls(directory,inventory,local_storage_limit_bytes=local_storage_limit_bytes)
