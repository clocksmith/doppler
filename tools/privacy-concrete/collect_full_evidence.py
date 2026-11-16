"""Construct a public CI artifact from a fixed public fixture; never upload raw logs or key/state files."""
import argparse
import json
from pathlib import Path
import re
import math

PUBLIC_FIXTURE='qwen05b-capital'
REPORT_FIELDS=['schema','planId','modelId','linearProjectionCount','encryptedTileCount','encryptedSteps',
               'matchesFullQuantizedGreedyReference','floatGreedyMatches','maxLogitErrors','setupMs','stepMs',
               'quantizedReferenceMs','elapsedMs','clientEncryptedReplayMs',
               'evaluatorRoundTripMs','sentBytes','receivedBytes','evaluationKeyBytes','clientPeakRss',
               'clientComputation','integrity']
PROJECTION=re.compile(r'(?:model\.layers\.(?:[0-9]|1[0-9]|2[0-3])\.(?:self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)|lm_head)\Z')


def collect(report_path,progress_path,revision):
    progress=[]
    if Path(progress_path).is_file():
        for line in Path(progress_path).read_text(errors='replace').splitlines():
            try:value=json.loads(line)
            except (ValueError,TypeError):continue
            if not isinstance(value,dict):continue
            if value.get('event')=='verified-tile' and type(value.get('sequence')) is int and PROJECTION.fullmatch(str(value.get('projection',''))):
                progress.append(dict(sequence=value['sequence'],projection=value['projection']))
    result=dict(schema='doppler.public-full-qwen-ci-evidence/v1',revision=revision,fixtureId=PUBLIC_FIXTURE,
                completed=False,productionBackendAvailable=False,securityQualified=False,
                verifiedTilesObserved=len(progress),lastVerifiedTile=progress[-1] if progress else None)
    if Path(report_path).is_file():
        try:report=json.loads(Path(report_path).read_text())
        except ValueError:return result
        if (report.get('publicFixtureId')==PUBLIC_FIXTURE and report.get('qualified') is False
                and report.get('productionBackendAvailable') is False and report.get('linearProjectionCount')==169
                and report.get('encryptedSteps')==4 and report.get('encryptedTileCount')==1548 and report.get('matchesFullQuantizedGreedyReference') is True):
            safe={}
            numeric_arrays={'floatGreedyMatches','maxLogitErrors','stepMs','quantizedReferenceMs'}
            for key in REPORT_FIELDS:
                value=report.get(key)
                if key=='schema' and value=='doppler.full-qwen-hybrid-execution/v1':
                    safe[key]=value
                elif key in {'planId','modelId'} and isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value):
                    safe[key]=value
                elif key in numeric_arrays and isinstance(value,list) and len(value)<=4 and all(isinstance(item,(int,float)) and math.isfinite(item) for item in value):
                    safe[key]=value
                elif type(value) in (int,float) and math.isfinite(value):
                    safe[key]=value
                elif key=='matchesFullQuantizedGreedyReference' and value is True:
                    safe[key]=True
            safe['clientComputation']='Client nonlinearities, embeddings, caches, quantization, decoding and full encrypted replay'
            safe['integrity']='Full deterministic replay; no qualified malicious-security claim'
            versions=report.get('implementationVersions',{})
            dependencies=['concrete-ml','concrete-ml-extensions','concrete-python','torch','transformers','numpy']
            safe['implementationVersions']={key:versions[key] for key in dependencies if isinstance(versions.get(key),str) and re.fullmatch(r'[0-9][0-9A-Za-z.+_-]{0,63}',versions[key])}
            result.update(completed=True,report=safe,
                          publicFixtureOutputText=report['clientLocalOutputText'],publicFixtureOutputTokens=report['clientLocalOutputTokens'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',required=True);parser.add_argument('--progress',required=True)
    parser.add_argument('--revision',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(collect(args.report,args.progress,args.revision),indent=2)+'\n')
