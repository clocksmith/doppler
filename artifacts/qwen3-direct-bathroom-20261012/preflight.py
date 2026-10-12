from pathlib import Path
import argparse,hashlib,json
from transformers import AutoTokenizer
from tokenizers import Tokenizer
parser=argparse.ArgumentParser();parser.add_argument('--doppler',required=True);parser.add_argument('--source',required=True);args=parser.parse_args()
root=Path(args.doppler);source=Path(args.source);here=Path(__file__).resolve().parent
suite=json.loads((here/'suite.json').read_text());base=json.loads((root/'artifacts/contractor-structured-quality-20261011/suite.json').read_text());policy=json.loads((root/'artifacts/qwen3-17b-application-policy-20261011/recommended-thinking-suite.json').read_text())
assert suite['cases']==base['cases'][:2]
for field in ['modelId','sourceRevision','enableThinking','generation','seeds','answerBoundary','maxOutputCharacters','eosTokenIds','officialPolicy']:assert suite[field]==base[field],field
for field in ['modelId','sourceRevision','enableThinking','generation','seeds','answerBoundary','maxOutputCharacters']:assert suite[field]==policy[field],field
assert 'instruction' not in suite and suite['seeds']==[17,53]
tok=AutoTokenizer.from_pretrained(source,local_files_only=True);fast=Tokenizer.from_file(str(source/'tokenizer.json'));rows=[]
for task in suite['cases']:
 messages=task['messages'];assert all(m['role']!='system' for m in messages)
 rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
 ids=tok.encode(rendered,add_special_tokens=False);assert ids==fast.encode(rendered,add_special_tokens=False).ids
 assert len(ids)+suite['generation']['maxTokens']<=suite['generation']['maxSeqLen']
 rows.append({'id':task['id'],'messages':messages,'inputTokenIds':ids,'inputTokens':len(ids),'renderedPromptSha256':hashlib.sha256(rendered.encode()).hexdigest(),'unchangedReviewCriteria':task['reviewCriteria']})
result={'schema':'doppler.direct-thinking-preflight/v1','sourceModelLoaded':False,'exactParentCaseObjects':True,'sameOfficialThinkingPolicy':True,'solePromptDelta':'Removal of added structured system instruction','sourceIdentity':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['download-receipt.json','tokenizer_config.json','tokenizer.json','generation_config.json']},'officialTemplateSha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),'pairs':4,'rows':rows}
(here/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pairs':4,'promptTokens':[r['inputTokens'] for r in rows],'exactCases':True,'samePolicy':True,'noModelLoaded':True}))
