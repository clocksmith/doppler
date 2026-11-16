"""Request-scoped actual Flash text input/output; local files, no shared private tokenizer cache."""
import json
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from protocol import require_active,POLICY,digest
from source_snapshot import pinned_source

SOURCE=json.loads(Path(__file__).with_name('flash-tokenizer.json').read_text())


def _load_tokenizer(directory):
    expected=dict(transformers=POLICY['dependencies']['transformers'],**SOURCE['additionalDependencies'])
    if any(version(name)!=value for name,value in expected.items()):raise ValueError('Pinned Flash tokenizer implementation required')
    # A fixed installed library class; downloaded Python and chat-template code never execute.
    from transformers import Qwen2TokenizerFast
    return Qwen2TokenizerFast.from_pretrained(directory,local_files_only=True)


def _limit(value):
    if type(value) is not int or not 0<value<=SOURCE['maxPositionEmbeddings']:raise ValueError('Explicit bounded Flash token limit required')


def _materialize(directory,cancelled):
    require_active(cancelled)
    config=json.loads((Path(directory)/'config.json').read_text())
    tokenizer_config=json.loads((Path(directory)/'tokenizer_config.json').read_text())
    if (config['text_config']['vocab_size']!=SOURCE['vocabSize'] or tokenizer_config.get('tokenizer_class')!=SOURCE['tokenizerClass']
            or tokenizer_config.get('auto_map') is not None):raise ValueError('Installed Flash tokenizer identity required')
    tokenizer=_load_tokenizer(directory)
    if tokenizer.is_fast is not True:raise ValueError('Inspected local fast tokenizer required')
    require_active(cancelled)
    return tokenizer


@contextmanager
def _client_tokenizer(directory,cancelled):
    require_active(cancelled)
    with pinned_source(directory,{name:record['sha256'] for name,record in SOURCE['files'].items()}):
        tokenizer=_materialize(directory,cancelled)
        try:yield tokenizer
        finally:tokenizer=None
    require_active(cancelled)


def encode(directory,prompt,max_tokens,*,cancelled=None):
    require_active(cancelled);_limit(max_tokens)
    if not isinstance(prompt,str):raise ValueError('Bounded client prompt required')
    try:size=len(prompt.encode('utf-8'))
    except UnicodeError:raise ValueError('Valid client UTF-8 prompt required') from None
    if not 0<size<=SOURCE['maxPromptBytes']:raise ValueError('Bounded client prompt required')
    with _client_tokenizer(directory,cancelled) as tokenizer:
        try:tokens=tokenizer.encode(prompt,add_special_tokens=False)
        except Exception:raise ValueError('Flash client tokenization rejected') from None
        require_active(cancelled)
        if not isinstance(tokens,list) or not 0<len(tokens)<=max_tokens or any(type(token) is not int or not 0<=token<SOURCE['vocabSize'] for token in tokens):
            raise ValueError('Flash prompt exceeds the admitted token contract')
        result=tuple(tokens)
    require_active(cancelled)
    return result


def decode(directory,tokens,max_tokens,*,cancelled=None):
    require_active(cancelled);_limit(max_tokens)
    if (not isinstance(tokens,(tuple,list)) or not 0<len(tokens)<=max_tokens
            or any(type(token) is not int or not 0<=token<SOURCE['vocabSize'] for token in tokens)):
        raise ValueError('Bounded client output tokens required')
    with _client_tokenizer(directory,cancelled) as tokenizer:
        try:result=tokenizer.decode(list(tokens),skip_special_tokens=False,clean_up_tokenization_spaces=False)
        except Exception:raise ValueError('Flash client token decoding rejected') from None
        require_active(cancelled)
        if not isinstance(result,str):raise ValueError('Bounded client output text required')
        try:size=len(result.encode('utf-8'))
        except UnicodeError:raise ValueError('Valid client UTF-8 output required') from None
        if size>SOURCE['maxOutputBytes']:raise ValueError('Bounded client output text required')
    require_active(cancelled)
    return result


if __name__=='__main__':
    import argparse
    import time
    from full_plan import canonical
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',required=True)
    parser.add_argument('--public-fixture',choices=[SOURCE['publicFixture']['id']],required=True);args=parser.parse_args()
    fixture=SOURCE['publicFixture'];began=time.perf_counter();tokens=encode(args.source_dir,fixture['text'],fixture['maxTokens'])
    result=decode(args.source_dir,tokens,fixture['maxTokens'])
    if result!=fixture['text']:raise ValueError('Actual public Flash tokenizer roundtrip mismatch')
    print(json.dumps(dict(schema='doppler.public-flash-client-text-io/v1',fixtureId=fixture['id'],sourceId=digest(canonical(SOURCE)),
        productionBackendAvailable=False,securityQualified=False,fullModelExecuted=False,encryptedExecution=False,
        promptTokens=list(tokens),roundtripText=result,elapsedMs=(time.perf_counter()-began)*1000,
        implementationVersions={name:version(name) for name in ['transformers','tokenizers']},
        ownership='Fresh local tokenizer per encode/decode call; no shared request-text cache',
        interpretation='One fixed public text fixture; no model output, encrypted inference, security or latency-distribution qualification'),indent=2))
