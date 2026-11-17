"""Actual native bit-equivalence and one-observation CPU costs; fixed public fixtures only."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import concrete_ml_extensions as ext
import concrete_ml_extensions.concrete_ml_extensions as binary

ROOT=Path(__file__).resolve().parent
POLICY=json.loads((ROOT/'build-policy.json').read_text())
sys.path.insert(0,str(ROOT.parent))


def run(lock_path,public_state):
    public_state.update(phase='implementation-admission')
    if version('concrete-ml-extensions')!=POLICY['nativeVersion'] or ext.is_cuda_enabled():raise ValueError('Exact CPU research build required')
    expected_build=hashlib.sha256(json.dumps(POLICY,sort_keys=True).encode()+(ROOT/'ntt_cpu.rs').read_bytes()+(ROOT/'python_api.rs').read_bytes()).hexdigest()
    if ext.__doppler_ntt_build_id__!=expected_build:raise ValueError('Native research source identity changed')
    base=json.loads((ROOT.parent/'policy.json').read_text())['parameters']
    if json.loads(ext.default_params())!=base:raise ValueError('Upstream cryptographic parameters changed')
    high=json.loads((ROOT.parent/'precision-policy.json').read_text())['parameters']
    quant12=json.loads((ROOT.parent/'precision12-policy.json').read_text())['parameters']
    from flash_compute_budget import workload
    flash_shapes=[(1,row['inputWidth'],row['outputWidth']) for row in workload()['shapes']]+[(5,2048,640)]
    generator=np.random.default_rng(23);cases=[];preparation=[]
    for profile,definition,geometries in [('upstream-default',base,[(1,16,2),(1,896,128),(1,2048,2048),(1,2048,2051),(2,16,2),(1,4098,3)]),
                                          ('declared-high-precision',high,[(1,2048,2),(1,2048,513)]),
                                          ('declared-quant12-message35',quant12,flash_shapes)]:
        public_state.update(phase='key-generation',profile=profile)
        parameters=ext.MatmulCryptoParameters.deserialize(json.dumps(definition));secret,key=ext.cpu_create_private_key(parameters)
        key_bytes=key.serialize();key_digest=hashlib.sha256(key_bytes).hexdigest()
        public_state.update(phase='key-precomputation')
        began=time.perf_counter();ext.cpu_prepare_ntt_key(key)
        preparation.append(dict(profile=profile,keyPrecomputeMs=(time.perf_counter()-began)*1000,evaluationKeyBytes=len(key_bytes)))
        for rows,width,columns in geometries:
            public_state.update(phase='matrix-evaluation',case=dict(rows=rows,inputWidth=width,outputWidth=columns))
            bits=12 if profile=='declared-quant12-message35' else 8
            values=generator.integers(0,2**bits,size=(rows,width),dtype=np.uint64)
            weights=generator.integers(-2**(bits-1),2**(bits-1),size=(width,columns),dtype=np.int64).astype(np.uint64)
            encrypted=ext.encrypt_matrix(pkey=secret,crypto_params=parameters,data=values)
            measured={};results=[]
            operations=[('original',ext.cpu_matrix_multiplication),('serial',ext.cpu_matrix_multiplication_ntt_serial),
                ('parallel-reference',ext.cpu_matrix_multiplication_ntt_parallel_reference),
                ('parallel-fused',ext.cpu_matrix_multiplication_ntt),('parallel-fused-replay',ext.cpu_matrix_multiplication_ntt)]
            # Alternate fused/reference order across public cases on the same key.
            if len(cases)%2:operations[2],operations[3]=operations[3],operations[2]
            for label,operation in operations:
                began=time.perf_counter();result=operation(encrypted,weights,key);serialized=result.serialize()
                measured[label]=(time.perf_counter()-began)*1000;results.append(serialized)
            public_state.update(phase='byte-equivalence')
            if any(value!=results[0] for value in results[1:]):raise ValueError('Native result bytes differ from the pinned original arithmetic')
            if hashlib.sha256(key.serialize()).hexdigest()!=key_digest:raise ValueError('Public cache changed serialized key material')
            cases.append(dict(profile=profile,rows=rows,inputWidth=width,outputWidth=columns,byteIdenticalToOriginal=True,
                              originalMs=measured['original'],serialNttMs=measured['serial'],parallelReferenceNttMs=measured['parallel-reference'],
                              nttMs=measured['parallel-fused'],nttReplayMs=measured['parallel-fused-replay'],
                              timingOrder=[label for label,_ in operations],resultBytes=len(results[0])))
        # Deserializing public evaluation material must create a fresh cache without wire changes.
        restored=ext.CpuCompressionKey.deserialize(key_bytes)
        if restored.serialize()!=key_bytes:raise ValueError('Evaluation-key wire compatibility changed')
        del restored,key,secret
    binary_path=Path(binary.__file__)
    return dict(schema='doppler.native-exact-ntt-experiment/v1',productionBackendAvailable=False,securityQualified=False,
                upstreamRevision=POLICY['revision'],nativeVersion=version('concrete-ml-extensions'),nativeBuildId=expected_build,
                binarySha256=hashlib.sha256(binary_path.read_bytes()).hexdigest(),
                cargoLockSha256=hashlib.sha256(Path(lock_path).read_bytes()).hexdigest(),
                nttCrate=POLICY['nttCrate'],keyPreparations=preparation,cases=cases,
                clientLifetimePeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
                interpretation='Fixed public native arithmetic fixtures, including full packing/multi-block/multi-row; one timing observation, no model throughput or independent security review')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lock',required=True);args=parser.parse_args()
    public_state={}
    try:report=run(args.lock,public_state)
    except BaseException:
        print(json.dumps(dict(schema='doppler.native-exact-ntt-experiment/v1',productionBackendAvailable=False,securityQualified=False,status='native-equivalence-or-build-check-failed',publicPhase=public_state)),flush=True)
        raise
    print(json.dumps(report,indent=2))
