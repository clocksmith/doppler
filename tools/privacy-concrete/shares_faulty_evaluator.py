"""A deliberately dishonest evaluator, for tests and the failure probe only. Never deploy it.

It serves exactly like shares_evaluator.py except that the answer to its N-th request is wrong: either one value
is changed (`flip`) or the whole answer is zero (`zeros`). Everything else, including every later answer, is honest,
so a run shows what the requester does after it has caught one failure.
"""
import argparse
import sys

import shares
import shares_evaluator


def install(fail_call,mode):
    if type(fail_call) is not int or fail_call<1 or mode not in ('flip','zeros'):
        raise ValueError('A positive request number and a mode of flip or zeros are required')
    honest=shares_evaluator.field_product
    state=dict(count=0)
    def dishonest(rows,weights,*args,**kwargs):
        state['count']+=1
        answer=honest(rows,weights,*args,**kwargs)
        if state['count']!=fail_call:return answer
        answer=answer.copy()
        if mode=='zeros':answer[:]=0
        else:answer[0,0]=(int(answer[0,0])+1)%shares.PRIME
        return answer
    shares_evaluator.field_product=dishonest
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--fail-call',type=int,required=True)
    parser.add_argument('--mode',choices=('flip','zeros'),default='flip')
    args=parser.parse_args()
    install(args.fail_call,args.mode)
    shares_evaluator.serve(args.cache_dir,sys.stdin.buffer,sys.stdout.buffer)
