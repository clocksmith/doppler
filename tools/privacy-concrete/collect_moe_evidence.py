"""Construct only fixed-public-fixture MoE evidence; never export raw client diagnostics."""
import argparse
import json
import math
from pathlib import Path
import re
from flash_moe_plan import CONFIG,PROGRAM,SOURCE_ID,lowered_operations
from full_plan import tiles
from full_status import FAILURE_CODES
from flash_moe_ntt_plan import CONFIG_NTT
from plan_backends import MOE_PROFILES
from native_runtime import ntt_source_id

SCHEDULE=tuple(row['name'] for row in lowered_operations() for _ in tiles(dict(row,weightId='0'*64),CONFIG['inputTileWidth'],CONFIG['outputTileWidth']))
PROFILE_CONFIGS={CONFIG['profile']:CONFIG,CONFIG_NTT['profile']:CONFIG_NTT}
HASH=re.compile(r'[0-9a-f]{64}\Z')
METRICS=('maxQuantizedVsFloatingError','maxEncryptedVsQuantizedError','executeMs','elapsedMs','evaluationKeyBytes','sentBytes','receivedBytes','clientEncryptedReplayMs','evaluatorRoundTripMs')


def collect(report_path,progress_path,revision,backend_profile=CONFIG['profile']):
    if not isinstance(backend_profile,str) or backend_profile not in PROFILE_CONFIGS:raise ValueError('Installed public MoE evidence profile required')
    policy=PROFILE_CONFIGS[backend_profile]
    observed=[];valid=True
    if Path(progress_path).is_file():
        for line in Path(progress_path).read_text(errors='replace').splitlines():
            try:event=json.loads(line)
            except (ValueError,TypeError):continue
            if not isinstance(event,dict) or event.get('event')!='verified-tile':continue
            index=len(observed)
            if (index>=len(SCHEDULE) or type(event.get('sequence')) is not int
                    or event['sequence']!=index+1 or event.get('projection')!=SCHEDULE[index]):valid=False;break
            observed.append(dict(sequence=index+1,projection=SCHEDULE[index]))
    result=dict(schema='doppler.public-flash-moe-ci-evidence/v1',revision=revision,fixtureId='flash-dense-moe-sine-2560',
                completed=False,productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,backendProfile=backend_profile,
                plannedEncryptedTiles=len(SCHEDULE),verifiedTilesObserved=len(observed),progressValid=valid,
                lastVerifiedTile=observed[-1] if observed else None)
    if not Path(report_path).is_file():return result
    try:report=json.loads(Path(report_path).read_text())
    except ValueError:return result
    if not isinstance(report,dict):return result
    if report.get('backendProfile',CONFIG['profile'])!=backend_profile:return result
    status=report.get('status')
    if isinstance(status,dict) and status.get('event')=='run-failed' and isinstance(status.get('failureCode'),str) and status['failureCode'] in FAILURE_CODES:
        result['failureCode']=status['failureCode'];return result
    exact=dict(schema='doppler.actual-flash-dense-moe-execution/v1',fixtureId=result['fixtureId'],
               productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,
               sourceId=SOURCE_ID,programId=PROGRAM.identity,protocol=policy['protocol'],expertCount=PROGRAM.experts.expert_count,
               topK=PROGRAM.experts.top_k,logicalProjections=len(PROGRAM.contract()['operations']),linearModules=len(lowered_operations()),
               plannedEncryptedTiles=len(SCHEDULE),encryptedTiles=len(SCHEDULE),allExpertsVisited=True,
               execution='encrypted',encryptedExecutionCompleted=True,quantizedReferenceCheckPassed=True)
    if (not valid or len(observed)!=len(SCHEDULE) or any(type(report.get(key)) is not type(value) or report[key]!=value for key,value in exact.items())
            or any(not isinstance(report.get(key),str) or not HASH.fullmatch(report[key]) for key in ['modelId','planId'])
            or any(type(report.get(key)) not in (int,float) or not math.isfinite(report[key]) or report[key]<0 for key in METRICS)):
        return result
    safe=dict(exact,**{key:report[key] for key in (*METRICS,'modelId','planId')})
    safe['backendProfile']=backend_profile
    if backend_profile==CONFIG_NTT['profile']:
        versions=report.get('implementationVersions');keys=['native-source-id','native-binary-sha256']
        if not isinstance(versions,dict) or any(not isinstance(versions.get(key),str) or not HASH.fullmatch(versions[key]) for key in keys) or versions['native-source-id']!=ntt_source_id():return result
        safe['nativeImplementation']={key:versions[key] for key in keys}
    result.update(completed=True,report=safe)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);parser.add_argument('--progress',required=True)
    parser.add_argument('--revision',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--backend-profile',choices=MOE_PROFILES,default=CONFIG['profile']);args=parser.parse_args()
    Path(args.output).write_text(json.dumps(collect(args.report,args.progress,args.revision,args.backend_profile),indent=2)+'\n')
