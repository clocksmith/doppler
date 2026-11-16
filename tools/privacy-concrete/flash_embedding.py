"""Actual public-token input embedding and independent safetensors lookup; not a model forward pass."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
import torch
from safetensors import safe_open
from flash_tokenizer import SOURCE,encode
from local_token_rows import LocalTokenRows,POLICY,installed_inventory
from local_token_embedding import LocalTokenEmbedding
from flash_model_state import FlashModelState
from flash_request_state import ClientContext
from flash_client_projections import client_execution
from full_plan import canonical
from protocol import digest


def run(source_dir,tokenizer_dir,local_storage_limit_bytes):
    began=time.perf_counter();spec=installed_inventory();source_id=digest(canonical(spec));fixture=SOURCE['publicFixture']
    storage=LocalTokenRows.flash(source_dir,local_storage_limit_bytes=local_storage_limit_bytes)
    state=None
    try:
        tokens=encode(tokenizer_dir,fixture['text'],fixture['maxTokens']);embedding=LocalTokenEmbedding(storage,max_lookup_rows=fixture['maxTokens'])
        context=ClientContext('public-fixture',source_id,source_id,'no-encryption-key','public-input-embedding');state=FlashModelState(context,(),(),source_id)
        with client_execution(state,context,(),'public-reference',None,None):
            actual=embedding(torch.tensor([tokens],dtype=torch.int64));state.complete(len(tokens))
        name=POLICY['tensorName'];file=spec['parts'][0]['shard']
        with safe_open(Path(source_dir)/file,framework='pt',device='cpu') as source:
            expected=torch.stack([source.get_slice(name)[index] for index in tokens]).float().unsqueeze(0)
        if not torch.equal(actual,expected):raise ValueError('Actual token embedding disagrees with independent safetensors lookup')
        return dict(schema='doppler.public-flash-input-embedding/v1',fixtureId=fixture['id'],sourceId=source_id,tokenizerSourceId=spec['tokenizerSourceId'],
                    productionBackendAvailable=False,securityQualified=False,fullModelExecuted=False,encryptedExecution=False,
                    promptTokens=list(tokens),hiddenShape=list(actual.shape),hiddenDtype=str(actual.dtype),hiddenSha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
                    independentReferenceEqual=True,wholeShardBytes=spec['wholeShardBytes'],selectedBF16Bytes=len(tokens)*storage.columns*2,
                    fullFP32EmbeddingBytes=storage.rows*storage.columns*4,elapsedMs=(time.perf_counter()-began)*1000,
                    clientLifetimePeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
                    interpretation='Actual checkpoint input embeddings only; complete-file verification and client text I/O included; no decoder/model output, crypto or latency-distribution qualification')
    finally:
        if state is not None:state.close()
        storage.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',required=True);parser.add_argument('--tokenizer-dir',required=True)
    parser.add_argument('--local-storage-limit-bytes',type=int,required=True);parser.add_argument('--public-fixture',choices=[SOURCE['publicFixture']['id']],required=True)
    args=parser.parse_args();print(json.dumps(run(args.source_dir,args.tokenizer_dir,args.local_storage_limit_bytes),indent=2))
