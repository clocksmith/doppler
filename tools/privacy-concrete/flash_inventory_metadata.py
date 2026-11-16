"""Bounded immutable-checkpoint metadata acquisition; never download weight payloads."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import re
import struct
import urllib.request
from protocol import parse_unique_json

def public_bytes(url,limit,expected=None):
    with urllib.request.urlopen(url,timeout=60) as response:
        if response.status!=200:raise ValueError('Complete bounded public metadata required')
        raw=response.read(limit+1)
    if len(raw)>limit or (expected is not None and hashlib.sha256(raw).hexdigest()!=expected):
        raise ValueError('Pinned bounded public metadata required')
    return raw



def model_metadata(base):
    origin='https://huggingface.co/'+base['repository']+'/resolve/'+base['revision']+'/'
    metadata={}
    for name,record in base['files'].items():
        raw=public_bytes(origin+name,4*1024*1024,record['sha256'])
        if len(raw)!=record['bytes']:raise ValueError('Pinned metadata length required')
        metadata[name]=parse_unique_json(raw)
    return metadata


def shard_headers(base,files):
    files=tuple(sorted(set(files)))
    if not files or any(not isinstance(name,str) or not re.fullmatch(r'model-[0-9]{5}-of-[0-9]{5}\.safetensors',name) for name in files):
        raise ValueError('Explicit checkpoint shard names required')
    origin='https://huggingface.co/'+base['repository']+'/resolve/'+base['revision']+'/'
    tree=parse_unique_json(public_bytes('https://huggingface.co/api/models/'+base['repository']+'/tree/'+base['revision']+'?recursive=false&expand=false',4*1024*1024))
    records={v['path']:dict(bytes=v['size'],sha256=v['lfs']['oid']) for v in tree if v.get('path') in files}
    if set(records)!=set(files) or any(type(v['bytes']) is not int or v['bytes']<=8 or not re.fullmatch(r'[0-9a-f]{64}',v['sha256']) for v in records.values()):
        raise ValueError('Complete pinned-revision object inventory required')
    def header(name):
        record=records[name]
        def part(begin,end):
            request=urllib.request.Request(origin+name,headers={'Range':f'bytes={begin}-{end}','Accept-Encoding':'identity'})
            with urllib.request.urlopen(request,timeout=60) as response:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {begin}-{end}/{record["bytes"]}':
                    raise ValueError('Exact bounded header range required; no full-weight fallback')
                raw=response.read(end-begin+2)
            if len(raw)!=end-begin+1:raise ValueError('Complete bounded header range required')
            return raw
        length=struct.unpack('<Q',part(0,7))[0]
        if not 0<length<=1024*1024 or length+8>record['bytes']:raise ValueError('Bounded public tensor header required')
        raw=part(8,7+length)
        return name,(dict(record,headerBytes=length,headerSha256=hashlib.sha256(raw).hexdigest()),parse_unique_json(raw))
    with ThreadPoolExecutor(max_workers=4) as pool:return dict(pool.map(header,files))
