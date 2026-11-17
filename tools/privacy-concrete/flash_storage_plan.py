"""Pinned source/whole-object disk planning; no downloads, model execution or admission.

Cache estimates conservatively include every staged BF16 parameter, including
nonlinear parameters that may not need a quantized cache; no deduplication saving
is assumed. Stage eviction is not implemented in the encrypted execution bridge.
"""
import argparse
import copy
import json
from pathlib import Path
from flash_source_scopes import TEXT,staging_spec
from local_token_rows import installed_inventory
from weight_encoding import encoding_for_bits
from flash_text_plan import GEOMETRY

TABLE=json.loads(Path(__file__).with_name('flash-ngram-storage.json').read_text())


def storage_plan(quantization_bits):
    encoding=encoding_for_bits(quantization_bits)
    coefficient_bytes=encoding['bytes']
    resident=copy.deepcopy(TABLE['shards'])
    for name,record in installed_inventory()['shards'].items():
        if name in resident and resident[name]!=record:raise ValueError('Public row-store object identities differ')
        resident[name]=record
    stages=[]
    for part,index in [('decoder',i) for i in range(TEXT['layerCount'])]+[('output',None)]:
        spec=staging_spec(part,index);objects=dict(resident)
        for name,record in spec['shards'].items():
            if name in objects and objects[name]!=record:raise ValueError('Public model-part object identities differ')
            objects[name]=record
        raw_bytes=sum(row['bytes'] for row in objects.values())+TEXT['auxiliarySourceBytes']
        cache_upper=sum((row['dataOffsets'][1]-row['dataOffsets'][0])//2*coefficient_bytes
                        for row in spec['tensors'].values() if row['dtype']=='BF16')
        stages.append(dict(part=part,layerIndex=index,wholePartShardBytes=spec['wholeShardBytes'],wholePartFiles=len(spec['shards']),
                           residentAndPartRawBytes=raw_bytes,coefficientCacheBytesUpperBound=cache_upper,
                           residentRawAndPartCacheBytesUpperBound=raw_bytes+cache_upper))
    return dict(schema='doppler.flash-public-storage-plan/v1',sourceRevision=TEXT['revision'],quantizationBits=quantization_bits,
                coefficientBytes=coefficient_bytes,completeSourceBytes=TEXT['wholeSourceBytes'],
                exactLinearCoefficientBytes=sum(width*height*coefficient_bytes for width,height in GEOMETRY.values()),
                largestLinearCoefficientBytes=max(width*height*coefficient_bytes for width,height in GEOMETRY.values()),
                persistentLocalRowStoreWholeBytes=sum(row['bytes'] for row in resident.values()),
                allPartCoefficientCacheBytesUpperBound=sum(row['coefficientCacheBytesUpperBound'] for row in stages),
                fullyResidentSourceAndCachesBytesUpperBound=TEXT['wholeSourceBytes']+sum(row['coefficientCacheBytesUpperBound'] for row in stages),
                stagedRawSourcePeakBytes=max(row['residentAndPartRawBytes'] for row in stages),
                stagedRawAndOneCachePeakBytesUpperBound=max(row['residentRawAndPartCacheBytesUpperBound'] for row in stages),stages=stages,
                excludes=['filesystem reserves and unrelated files','client/evaluator native build products and dependencies','native keys, result/activation buffers and runtime memory','provider storage'],
                requirements=['Owned whole-object source staging is implemented separately; complete PLE/input row stores remain mandatory',
                              'Encrypted bridge still requires full public coefficient-cache admission; per-stage cache eviction/provider loading is unfinished',
                              'Space estimates qualify no model precision, performance, platform, endpoint or cryptographic security'],
                downloadsPerformed=False,weightsMaterialized=False,fullFlashModelExecuted=False,productionBackendAvailable=False,securityQualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--quantization-bits',type=int,choices=[8,12],required=True);args=parser.parse_args()
    print(json.dumps(storage_plan(args.quantization_bits),indent=2))
