"""Physical local HTTP checks: identity, chat, stream, follow-up, cancel/reuse.

Usage: python3 probe-local-server.py BASE_URL OUTPUT
This is an operational smoke, never a numerical qualification replacement.
"""
import datetime
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

base,output=sys.argv[1],Path(sys.argv[2])
report=dict(scope='Actual local Flash-Next HTTP operations; long-context gate remains open',
            startedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseUrl=base,rows=[],fullModelQualified=False)


def require(condition,message):
    if not condition: raise ValueError(message)


def get(route):
    with urllib.request.urlopen(base+route,timeout=120) as response:return json.load(response)


def request(messages,stream=False,max_tokens=8):
    body=dict(model=model,messages=messages,stream=stream,max_tokens=max_tokens,temperature=0)
    return urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})


def chat(messages,expected):
    with urllib.request.urlopen(request(messages),timeout=120) as response:r=json.load(response)
    require(re.fullmatch(expected,r['choices'][0]['message']['content'].strip(),re.I),'Incorrect short chat answer')
    usage=r['usage']
    require(usage['prompt_tokens']>0 and usage['completion_tokens']>0 and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'],'Invalid measured usage')
    return dict(content=r['choices'][0]['message']['content'],usage=usage)


try:
    health=get('/health');model=health['model'];report['health']=health
    require(health['ready'] and 'gate remains open' in health['qualification'],'Missing honest ready/qualification state')
    models=get('/v1/models')['data']
    require(len(models)==1 and models[0]['id']==model and models[0]['doppler']['runtimePromotionState']=='experimental','Incorrect local model identity')
    report['rows'].append(dict(case='identity',passed=True))
    capital=[dict(role='user',content='What is the capital of France? Reply with only the city name.')]
    report['rows'].append(dict(case='chat',passed=True,**chat(capital,r'Paris[.!]?')))
    content='';done=False
    arithmetic=[dict(role='user',content='What is 17 + 25? Reply with only the number.')]
    with urllib.request.urlopen(request(arithmetic,True),timeout=120) as response:
        for line in response:
            if not line.startswith(b'data: '):continue
            value=line[6:].strip()
            if value==b'[DONE]':done=True;break
            content+=json.loads(value)['choices'][0]['delta'].get('content','')
    require(done and re.fullmatch(r'42[.!]?',content.strip()),'Incorrect or incomplete stream')
    report['rows'].append(dict(case='stream',passed=True,content=content))
    followup=capital+[dict(role='assistant',content='Paris'),dict(role='user',content='Which country is that city in? Reply with only the country name.')]
    report['rows'].append(dict(case='follow-up',passed=True,**chat(followup,r'France[.!]?')))
    first=urllib.request.urlopen(request([dict(role='user',content='Count from one to one hundred in words.')],True,128),timeout=120)
    try:
        try:
            urllib.request.urlopen(request(arithmetic),timeout=120)
            raise ValueError('Concurrent request was not refused')
        except urllib.error.HTTPError as error:require(error.code==503,'Unexpected concurrency status')
        observed=False
        for line in first:
            if line.startswith(b'data: ') and line[6:].strip()!=b'[DONE]':
                chunk=json.loads(line[6:])
                if chunk['choices'][0]['delta'].get('content'):observed=True;break
        require(observed,'Cancellation did not reach an active generation')
    finally:first.close()
    deadline=time.monotonic()+30
    while get('/health')['active']:
        require(time.monotonic()<deadline,'Cancelled request did not drain')
        time.sleep(0.1)
    report['rows'].append(dict(case='concurrent-refusal-and-active-client-cancel',passed=True))
    report['rows'].append(dict(case='reuse-after-cancel',passed=True,**chat(arithmetic,r'42[.!]?')))
    try:
        urllib.request.urlopen(request(arithmetic,max_tokens=129),timeout=120)
        raise ValueError('Output cap was not enforced')
    except urllib.error.HTTPError as error:require(error.code==400,'Output cap must be an actionable request error')
    report['rows'].append(dict(case='output-cap',passed=True))
    report['completed']=True
except Exception as error:
    report['failure']=str(error)
    raise
finally:
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(completed=report.get('completed',False),rows=report['rows'],failure=report.get('failure'))))
