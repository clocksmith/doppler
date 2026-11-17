"""Installed complete input-embedding layout over the shared client-local BF16 row owner."""
import json
from pathlib import Path
from local_bf16_rows import LocalBF16Rows
from flash_source_scopes import TEXT
from flash_tokenizer import SOURCE as TOKENIZER
from full_plan import canonical
from protocol import digest

POLICY=json.loads(Path(__file__).with_name('flash-embedding-policy.json').read_text())


def installed_inventory():
    value=TEXT['tensors'][POLICY['tensorName']];rows,columns=value['shape'];name=value['shard'];record=TEXT['shards'][name]
    if value['dtype']!='BF16' or rows!=TOKENIZER['vocabSize'] or TEXT['tokenizerSourceId']!=digest(canonical(TOKENIZER)):
        raise ValueError('Installed matching tokenizer and BF16 input embedding required')
    return dict(schema=POLICY['storageSchema'],sourceId=digest(canonical(TEXT)),tokenizerSourceId=TEXT['tokenizerSourceId'],
                logicalShape=[rows,columns],logicalBytes=rows*columns*2,wholeShardBytes=record['bytes'],shards={name:record},
                parts=[dict(name=POLICY['tensorName'],shard=name,firstRow=0,rows=rows,columns=columns,dtype='BF16',dataOffsets=value['dataOffsets'])])


class LocalTokenRows(LocalBF16Rows):
    def __init__(self,directory,inventory,*,local_storage_limit_bytes,cancelled=None):
        super().__init__(directory,inventory,local_storage_limit_bytes=local_storage_limit_bytes,layout_schema=POLICY['storageSchema'],cancelled=cancelled)

    @classmethod
    def flash(cls,directory,*,local_storage_limit_bytes,cancelled=None):return cls(directory,installed_inventory(),local_storage_limit_bytes=local_storage_limit_bytes,cancelled=cancelled)


