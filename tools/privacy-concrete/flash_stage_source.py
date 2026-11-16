"""Owned whole-object staging for one public decoder/output part, without private inputs.

Stages a small source subset under a distinct identity; no production backend or
full Flash execution is added. PLE rows remain with the separate complete local
row-store contract. No model/precision/recipient fallback is permitted.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from flash_source_scopes import staging_spec
from flash_weights import FlashSource,provision_spec


class FlashStageSource(FlashSource):
    def __init__(self,directory,part,layer_index=None):
        self.closed=False
        self._open(directory,staging_spec(part,layer_index))

    def tensor(self,name,expert=None):
        if self.closed:raise ValueError('Public model-part stage is closed')
        return super().tensor(name,expert)

    def close(self):self.closed=True


@contextmanager
def staged_source(parent_directory,part,layer_index=None):
    specification=staging_spec(part,layer_index) # Reject before filesystem/network work.
    with tempfile.TemporaryDirectory(dir=Path(parent_directory),prefix='flash-public-stage-') as temporary:
        provision_spec(temporary,specification)
        source=FlashStageSource(temporary,part,layer_index)
        try:yield source
        finally:source.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True)
    parser.add_argument('--part',choices=['decoder','output'],required=True);parser.add_argument('--layer-index',type=int)
    args=parser.parse_args();spec=staging_spec(args.part,args.layer_index)
    provision_spec(args.destination,spec)
    FlashStageSource(args.destination,args.part,args.layer_index).close()
    print(json.dumps(dict(schema=spec['schema'],part=spec['part'],layerIndex=spec['layerIndex'],parentSourceId=spec['parentSourceId'],
        completeShardBytes=spec['wholeShardBytes'],verified=True,completeTextModel=False,productionBackendAvailable=False)))
