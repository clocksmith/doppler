"""One installed public-prompt registry for clear quality and encrypted evidence."""
import json
from pathlib import Path
from types import MappingProxyType

SPEC=json.loads(Path(__file__).with_name('quantization-fixtures.json').read_text())
DEFAULT_PUBLIC_FIXTURE='qwen05b-capital'


def public_fixture_prompts(spec):
    if not isinstance(spec,dict) or spec.get('schema')!='doppler.qwen-public-quantization-fixtures/v1' or not isinstance(spec.get('fixtures'),list):
        raise ValueError('Installed public Qwen fixtures required')
    result={}
    for row in spec['fixtures']:
        if (not isinstance(row,dict) or set(row)!={'id','prompt','publicFixtureId'}
                or not isinstance(row['id'],str) or not 0<len(row['id'])<=64
                or not isinstance(row['publicFixtureId'],str) or not row['publicFixtureId'].startswith('qwen05b-') or len(row['publicFixtureId'])>80
                or row['publicFixtureId'] in result or not isinstance(row['prompt'],str) or not 0<len(row['prompt'].encode())<=4096):
            raise ValueError('Explicit unique public Qwen fixture identity/prompt required')
        result[row['publicFixtureId']]=row['prompt']
    if DEFAULT_PUBLIC_FIXTURE not in result:raise ValueError('Legacy public Qwen fixture required')
    return MappingProxyType(result)


PUBLIC_FIXTURES=public_fixture_prompts(SPEC)
