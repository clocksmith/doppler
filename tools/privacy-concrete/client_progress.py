"""Optional client-local counts, constructed without private execution evidence.

Transport progress is not acceptance. No integrity result is recorded until the
caller has closed the route and completed numerical validation. This does not
promise constant-time execution or remove observable stopping-position leakage.
"""
import json
import os
from pathlib import Path


class ClientProgress:
    def __init__(self,path):
        self.path=Path(path).absolute()
        descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        self.handle=os.fdopen(descriptor,'w',encoding='utf-8')
        self.plan_id=None;self.total=self.count=0;self.finished=False

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.handle.close()

    def _require(self,condition):
        if self.handle.closed or self.finished or not condition:raise ValueError('Client progress lifetime or route differs')

    def _emit(self,phase):
        # Construct the audience-specific record directly. Neither observer
        # dictionaries nor exceptions are serialized and later redacted.
        self.handle.write(json.dumps(dict(schema='doppler.client-route-progress/v1',audience='client-local',
            phase=phase,planId=self.plan_id,receivedTiles=self.count,plannedTiles=self.total,
            productionBackendAvailable=False,securityQualified=False),separators=(',',':'))+'\n')
        self.handle.flush()

    def bind(self,plan_id,total):
        self._require(self.plan_id is None and isinstance(plan_id,str) and len(plan_id)==64
            and all(c in '0123456789abcdef' for c in plan_id) and type(total) is int and 0<total<2**31)
        self.plan_id,self.total=plan_id,total;self._emit('planned')

    def observe(self,kind,value):
        self._require(self.plan_id is not None and kind in ('received-tile','verified-tile') and type(value) is dict)
        sequence=value.get('sequence')
        self._require(type(sequence) is int and sequence==self.count+1 and sequence<=self.total)
        self.count=sequence
        if sequence%256==0 or sequence==self.total:self._emit('receiving')

    def finish(self,*,prepare_only,readiness_only=False):
        self._require(self.plan_id is not None and type(prepare_only) is bool and type(readiness_only) is bool
            and not (prepare_only and readiness_only) and self.count==(0 if prepare_only or readiness_only else self.total))
        self._emit('prepared' if prepare_only else 'provider-ready' if readiness_only else 'validated');self.finished=True
