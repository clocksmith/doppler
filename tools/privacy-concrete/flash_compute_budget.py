"""Installed Flash workload inventory and explicitly synthetic kernel estimates."""
from collections import Counter
import json
import math
from full_plan import canonical,ranges
from protocol import digest
from flash_source_scopes import TEXT
from flash_text_plan import GEOMETRY
from flash_text_rebuild5_plan import CONFIG


def workload():
    histogram=Counter((end-begin,stop-start) for width,height in GEOMETRY.values()
        for begin,end in ranges(width,CONFIG['inputTileWidth']) for start,stop in ranges(height,CONFIG['outputTileWidth']))
    return dict(schema='doppler.installed-flash-workload/v1',sourceId=digest(canonical(TEXT)),backendProfile=CONFIG['profile'],
        projections=len(GEOMETRY),steps=CONFIG['steps'],prefillRows=CONFIG['contextBucket'],tilesPerStep=sum(histogram.values()),
        shapes=[dict(inputWidth=width,outputWidth=height,tilesPerStep=count) for (width,height),count in sorted(histogram.items())],
        logicalCoefficientBytesPerStep=sum(width*height*2 for width,height in GEOMETRY.values()),
        largestCoefficientBytes=max(width*height*2 for width,height in GEOMETRY.values()),
        logicalQuantizerMetadataBytes=sum(height for _,height in GEOMETRY.values())*3*4,
        quantizerMetadataAssumption='three float32 values per output channel; excludes graph, proofs, caches, keys and allocator overhead',
        productionBackendAvailable=False,securityQualified=False)


def project_native_cost(cases):
    """Weight measured synthetic evaluator calls; never call this model latency."""
    installed=workload();expected={(step,row['inputWidth'],row['outputWidth']) for step in range(installed['steps']) for row in installed['shapes']}
    observed={}
    for row in cases:
        if (not isinstance(row,dict) or any(type(row.get(name)) is not int for name in ('step','rows','inputWidth','outputWidth'))
                or 'nativeEvaluateSeconds' not in row):raise ValueError('Typed public kernel measurement required')
        key=(row['step'],row['inputWidth'],row['outputWidth']);seconds=row['nativeEvaluateSeconds']
        if (key not in expected or key in observed or row['rows']!=(installed['prefillRows'] if row['step']==0 else 1)
                or type(seconds) not in (float,int) or not math.isfinite(seconds) or seconds<0):
            raise ValueError('Complete unique installed synthetic geometry measurements required')
        observed[key]=seconds
    if set(observed)!=expected:raise ValueError('Every installed geometry and fixed step must be measured')
    seconds=sum(row['tilesPerStep']*observed[step,row['inputWidth'],row['outputWidth']]
        for step in range(installed['steps']) for row in installed['shapes'])
    return dict(schema='doppler.synthetic-flash-kernel-work-projection/v1',installedWorkloadId=digest(canonical(installed)),
        installedTiles=installed['tilesPerStep']*installed['steps'],weightedSyntheticNativeEvaluatorSeconds=seconds,
        measuredFullModelLatency=False,productionBackendAvailable=False,securityQualified=False,
        exclusions=['actual trained coefficients and activations','client encryption, decryption, checks and replay',
            'client nonlinear/model/reference execution','real model source verification, quantization and cache I/O',
            'transport, scheduling, different hardware and concurrency'])


if __name__=='__main__':print(json.dumps(workload(),indent=2))
