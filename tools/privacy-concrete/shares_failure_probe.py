"""Measure the open exception in Claim 4 of SHARES.md.

After the requester catches a wrong answer it substitutes zeros and carries on. If a later client step could then
fail to proceed, the route would close early and the number of calls a provider sees before that could depend on
the private input. This probe makes one provider answer one request wrongly, for several positions of that request
and several public prompts, and records what each provider was sent. The transcript is input-independent on these
cases if, for every position, the calls, flights and bytes are the same for every prompt and the same as an honest
run's. It measures counts and bytes only. It says nothing about timing.

Usage: python shares_failure_probe.py --model-dir M --cache-dir C [--public-fixture ID ...] [--fail-call N ...]
           [--mode flip|zeros] [--group-flights] > probe.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

from full_status import ExecutionRejected
from public_qwen_fixtures import PUBLIC_FIXTURES


def observe(model_dir,cache_dir,fixture,fail_call,mode,group_flights):
    """One attempt. `fail_call` None is an honest run. Returns what the two providers were sent and the outcome."""
    import shares_model  # needs the model toolchain; verdict() below does not
    here=Path(__file__).parent
    honest=[sys.executable,str(here/'shares_evaluator.py'),'--cache-dir',str(cache_dir)]
    faulty=honest if fail_call is None else [sys.executable,str(here/'shares_faulty_evaluator.py'),'--cache-dir',str(cache_dir),
                                             '--fail-call',str(fail_call),'--mode',mode]
    routes=[]
    class Recording(shares_model.ShareRoute):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);routes.append(self)
    original=shares_model.ShareRoute
    shares_model.ShareRoute=Recording
    steps=[]
    began=time.perf_counter();code=None
    try:
        shares_model.run(model_dir,cache_dir,PUBLIC_FIXTURES[fixture],[honest,faulty],group_flights=group_flights,
                         progress=lambda event:steps.append(event.get('sharedProjectionCalls')))
    except ExecutionRejected as error:
        code=error.code
    finally:
        shares_model.ShareRoute=original
    report=routes[-1].report()
    return dict(fixture=fixture,failCall=fail_call,mode=None if fail_call is None else mode,groupedFlights=bool(group_flights),
                outcome=code or 'completed',integrityFailureRecorded=routes[-1].integrity_failed(),
                projectionCalls=report['sharedProjectionCalls'],flights=report['sharedFlights'],rows=report['sharedRows'],
                sentBytes=report['sentBytes'],receivedBytes=report['receivedBytes'],callsAfterEachStep=steps,
                wallMs=(time.perf_counter()-began)*1000)


def transcript(row):
    return (row['projectionCalls'],row['flights'],row['rows'],tuple(row['sentBytes']),tuple(row['receivedBytes']),tuple(row['callsAfterEachStep']))


def verdict(rows):
    """Compare every dishonest attempt with the honest attempts. Pure, so it can be tested without a model."""
    honest=[row for row in rows if row['failCall'] is None]
    dishonest=[row for row in rows if row['failCall'] is not None]
    if not honest or not dishonest:raise ValueError('The probe needs at least one honest and one dishonest attempt')
    shapes={transcript(row) for row in honest}
    differing=[dict(fixture=row['fixture'],failCall=row['failCall'],mode=row['mode']) for row in dishonest if transcript(row) not in shapes]
    uncaught=[dict(fixture=row['fixture'],failCall=row['failCall'],mode=row['mode']) for row in dishonest
              if row['outcome']!='integrity-check-failed' or not row['integrityFailureRecorded']]
    unclean=[dict(fixture=row['fixture']) for row in honest if row['outcome']!='completed' or row['integrityFailureRecorded']]
    return dict(honestAttempts=len(honest),dishonestAttempts=len(dishonest),honestTranscriptShapes=len(shapes),
                attemptsWhoseTranscriptDiffersFromHonest=differing,dishonestAttemptsNotReportedAsIntegrityFailure=uncaught,
                honestAttemptsThatDidNotComplete=unclean,
                transcriptIndependentOfInputAndFailurePosition=len(shapes)==1 and not differing and not uncaught and not unclean,
                measures='projection calls, flights, rows, bytes sent and received per provider, and calls after each step; not timing')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',required=True);parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--public-fixture',action='append',choices=tuple(PUBLIC_FIXTURES))
    parser.add_argument('--fail-call',action='append',type=int)
    parser.add_argument('--mode',choices=('flip','zeros'),default='flip')
    parser.add_argument('--group-flights',action='store_true')
    args=parser.parse_args()
    fixtures=args.public_fixture or list(PUBLIC_FIXTURES)
    positions=args.fail_call or [1,2,85,169,170,400,676]
    rows=[]
    for fixture in fixtures:
        for position in [None]+positions:
            rows.append(observe(args.model_dir,args.cache_dir,fixture,position,args.mode,args.group_flights))
            print(json.dumps({key:rows[-1][key] for key in ('fixture','failCall','outcome','projectionCalls','flights')}),file=sys.stderr,flush=True)
    print(json.dumps(dict(schema='doppler.two-server-shares-failure-probe/v1',fixtures=fixtures,failCalls=positions,mode=args.mode,
                          groupedFlights=bool(args.group_flights),verdict=verdict(rows),attempts=rows),indent=2))
