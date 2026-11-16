"""Synthetic framing-only subprocess for tests; no model, cryptography or backend."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from full_plan import FullPlan
from protocol import receive,send,check,digest

plan=FullPlan(bytes.fromhex(sys.argv[1]));trace=Path(sys.argv[2]);events=[]
source,target=sys.stdin.buffer.raw,sys.stdout.buffer.raw
header,payload,_=receive(source,timeout=5)
session,key_id=header['session'],header['keyId'];check(header,plan.frame(session,key_id,0,'evaluation-key'))
if payload!=b'public-evaluation-fixture' or digest(payload)!=key_id:raise ValueError('Synthetic key fixture required')
events.append(header['kind']);trace.write_text(json.dumps(events))
send(target,plan.frame(session,key_id,0,'ready'))
for step in range(plan.definition()['steps']):
    tile=plan.schedule()[0];header,payload,_=receive(source,timeout=5)
    check(header,plan.frame(session,key_id,step+1,'evaluate',tile['id']))
    if payload!=b'ciphertext-only':raise ValueError('Synthetic opaque fixture required')
    events.append(header['kind']);trace.write_text(json.dumps(events))
    send(target,plan.frame(session,key_id,step+1,'result',tile['id']),b'valid-circuit-output')
header,payload,_=receive(source,timeout=5);check(header,plan.frame(session,key_id,5,'finish'))
if payload:raise ValueError('Empty finish required')
events.append(header['kind']);trace.write_text(json.dumps(events))
