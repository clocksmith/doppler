"""One evaluator for the two-server shares route. Needs only NumPy and the public coefficient files.

It multiplies whatever rows it is sent by the named public matrix modulo the prime 2**61 - 1. The rows are
uniformly random numbers, so this process handles no private data and needs no key.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from shares import CLOSE,POLICY,REQUEST,REQUEST_HEAD,RESPONSE,RESPONSE_HEAD,field_product


def serve(directory,source,sink):
    manifest=json.loads((Path(directory)/'shares-manifest.json').read_text())
    if manifest.get('schema')!='doppler.two-server-shares-manifest/v1':raise ValueError('Versioned shares manifest required')
    projections={row['name']:row for row in manifest['projections']}
    opened={}
    def matrix(name):
        if name not in opened:
            row=projections[name]
            if Path(row['file']).name!=row['file']:raise ValueError('Coefficient file must sit directly in the cache directory')
            path=Path(directory)/row['file']
            if path.is_symlink() or not path.is_file():raise ValueError('Regular public coefficient file required')
            mapped=np.memmap(path,dtype=row['dtype'],mode='r',shape=(row['inputWidth'],row['outputWidth']))
            # Identity is taken once over the mapped bytes. A later change to the file is the provider's own doing
            # and is caught by the requester's check, not here.
            if hashlib.sha256(mapped).hexdigest()!=row['weightId']:raise ValueError('Public coefficient identity changed')
            opened[name]=mapped
        return opened[name]
    def read(count):
        data=source.read(count)
        if len(data)!=count:raise EOFError
        return data
    while True:
        try:magic=read(4)
        except EOFError:return
        if magic==CLOSE:return
        if magic!=REQUEST:
            sink.write(RESPONSE_HEAD.pack(RESPONSE,0,1,0,0));sink.flush();return
        _,sequence,name_bytes,rows,width=REQUEST_HEAD.unpack(magic+read(REQUEST_HEAD.size-4))
        try:name=read(name_bytes).decode() if name_bytes<=1024 else None
        except UnicodeDecodeError:name=None
        if (name not in projections or width!=projections[name]['inputWidth']
                or not 1<=rows<=(1+2*POLICY['checkRounds'])*POLICY['maximumRows']):
            sink.write(RESPONSE_HEAD.pack(RESPONSE,sequence,1,0,0));sink.flush();return
        shares=np.frombuffer(read(8*rows*width),dtype='<u8').reshape(rows,width)
        answer=field_product(shares,matrix(name))
        sink.write(RESPONSE_HEAD.pack(RESPONSE,sequence,0,answer.shape[0],answer.shape[1]))
        sink.write(answer.astype('<u8',copy=False).tobytes());sink.flush()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache-dir',required=True)
    serve(parser.parse_args().cache_dir,sys.stdin.buffer,sys.stdout.buffer)
