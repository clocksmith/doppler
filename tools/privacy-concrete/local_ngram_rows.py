"""Client-local BF16 row storage. No networking, selected-row fetches or request cache.

The installed Flash inventory records a source-backed row layout, not provisioned
weights or production model qualification. Complete shards must verify before any
private row is accepted. Storage access patterns are visible to the trusted client
host/filesystem; this is not PIR or protection from that host.
"""
from bisect import bisect_right
import copy
import json
import mmap
from pathlib import Path
from flash_weights import verify_blob,file_signature
from protocol import parse_unique_json,require_active


class LocalNgramRows:
    def __init__(self,directory,inventory,*,local_storage_limit_bytes):
        self.handles={};self.maps={};self.signatures={};self.closed=False
        try:
            self.spec=copy.deepcopy(inventory);self.directory=Path(directory).resolve()
            self._validate(local_storage_limit_bytes)
            # Verify all public material before accepting indices; never acquire on cache miss.
            for name,record in self.spec['shards'].items():
                path=self.directory/name;before=file_signature(path);verify_blob(path,record)
                handle=path.open('rb');self.handles[name]=handle
                raw=handle.read(8)
                if len(raw)!=8 or int.from_bytes(raw,'little')!=record['headerBytes']:raise ValueError('Pinned local table header required')
                header=handle.read(record['headerBytes'])
                import hashlib
                if hashlib.sha256(header).hexdigest()!=record['headerSha256']:raise ValueError('Pinned local table header required')
                fields=parse_unique_json(header)
                for part in self.parts:
                    if part['shard']==name and fields.get(part['name'])!={key:part[key] for key in ['dtype','shape','data_offsets']}:
                        raise ValueError('Pinned local table offsets required')
                if file_signature(path)!=before:raise ValueError('Local table changed during verification')
                self.maps[name]=mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ);self.signatures[name]=before
            self._unchanged()
        except BaseException:
            self.close();raise

    @classmethod
    def flash(cls,directory,*,local_storage_limit_bytes):
        spec=json.loads(Path(__file__).with_name('flash-ngram-storage.json').read_text())
        return cls(directory,spec,local_storage_limit_bytes=local_storage_limit_bytes)

    def _validate(self,limit):
        s=self.spec
        if (type(limit) is not int or limit<=0 or not isinstance(s,dict)
                or s.get('schema')!='doppler.flash-local-ngram-storage-inventory/v1'
                or not isinstance(s.get('shards'),dict) or not s['shards']
                or not isinstance(s.get('parts'),list) or not s['parts']):raise ValueError('Explicit local table contract required')
        if s['wholeShardBytes']!=sum(item['bytes'] for item in s['shards'].values()) or s['wholeShardBytes']>limit:
            raise ValueError('Complete local ngram storage exceeds the declared budget')
        row=0;parts=[];names=set();used=set()
        for part in s['parts']:
            name=part['shard'];record=s['shards'][name]
            if (not isinstance(name,str) or Path(name).name!=name or name in ['.','..']
                    or any(type(part[key]) is not int for key in ['firstRow','rows','columns'])
                    or part['firstRow']!=row or part['rows']<=0 or part['columns']!=s['logicalShape'][1]
                    or part['dtype']!='BF16' or part['name'] in names
                    or type(record['bytes']) is not int or record['bytes']<=0
                    or type(record['headerBytes']) is not int or not 0<record['headerBytes']<=16*1024*1024):
                raise ValueError('Contiguous explicit BF16 row layout required')
            start,end=part['dataOffsets']
            if (type(start) is not int or type(end) is not int or start<0
                    or end-start!=part['rows']*part['columns']*2 or end+8+record['headerBytes']>record['bytes']):
                raise ValueError('Exact complete local row offsets required')
            parts.append(dict(part,shape=[part['rows'],part['columns']],data_offsets=[start,end]))
            names.add(part['name']);used.add(name);row+=part['rows']
        if row!=s['logicalShape'][0] or s['logicalBytes']!=row*s['logicalShape'][1]*2 or used!=set(s['shards']):
            raise ValueError('Complete local ngram layout required')
        self.parts=tuple(parts);self.starts=tuple(p['firstRow'] for p in self.parts)
        self.rows,self.columns=s['logicalShape']

    def _unchanged(self):
        if self.closed:raise ValueError('Local ngram storage is closed')
        for name,identity in self.signatures.items():
            if file_signature(self.directory/name)!=identity:raise ValueError('Verified local ngram storage changed')

    def read_rows(self,indices,*,max_output_bytes,cancelled=None):
        """Return owned little-endian BF16 bytes; do not retain indices or results."""
        try:
            require_active(cancelled);self._unchanged()
            if (not isinstance(indices,(list,tuple)) or not indices or type(max_output_bytes) is not int
                    or max_output_bytes<=0 or len(indices)*self.columns*2>max_output_bytes
                    or any(type(index) is not int or not 0<=index<self.rows for index in indices)):
                raise ValueError('Bounded local row indices required')
            result=bytearray()
            for index in indices:
                require_active(cancelled)
                part=self.parts[bisect_right(self.starts,index)-1];record=self.spec['shards'][part['shard']]
                offset=8+record['headerBytes']+part['dataOffsets'][0]+(index-part['firstRow'])*self.columns*2
                result.extend(self.maps[part['shard']][offset:offset+self.columns*2])
            require_active(cancelled);self._unchanged();return bytes(result)
        except BaseException:
            self.close();raise

    def close(self):
        self.closed=True
        for mapping in self.maps.values():mapping.close()
        for handle in self.handles.values():handle.close()
        self.maps.clear();self.handles.clear()

    def __enter__(self):self._unchanged();return self
    def __exit__(self,*args):self.close()
