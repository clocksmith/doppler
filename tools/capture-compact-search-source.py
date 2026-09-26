#!/usr/bin/env python3
"""CPU source screening only; no Doppler execution or release qualification."""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture(policy):
    torch.set_num_threads(policy['threads'])
    torch.manual_seed(0)
    inputs_path = Path(policy['preparedPath'])
    inputs = json.loads(inputs_path.read_text())
    sources = json.loads(Path(policy['sourcesPath']).read_text())
    for source in sources:
        root = Path(source['directory'])
        for entry in source['files']:
            metadata = root / '.cache/huggingface/download' / (entry['path'] + '.metadata')
            if metadata.read_text().splitlines()[0] != source['revision']:
                raise ValueError('Unpinned source file: ' + entry['path'])
            entry['sha256'] = digest(root / entry['path'])
    embedding_source = next(source for source in sources if source['role'] == 'embedding')
    reranker_source = next(source for source in sources if source['role'] == 'reranker')
    if embedding_source['repository'] != 'sentence-transformers/all-MiniLM-L6-v2':
        raise ValueError('This source capture implements MiniLM mean/L2 pooling only')
    if reranker_source['repository'] != 'cross-encoder/ms-marco-MiniLM-L6-v2':
        raise ValueError('This source capture implements the MiniLM scalar-logit head only')
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(embedding_source['directory'], local_files_only=True)
    model = AutoModel.from_pretrained(embedding_source['directory'], local_files_only=True,
                                     dtype=torch.float32, attn_implementation='eager').eval()
    rank_tokenizer = AutoTokenizer.from_pretrained(reranker_source['directory'], local_files_only=True)
    rank_model = AutoModelForSequenceClassification.from_pretrained(reranker_source['directory'],
        local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval()
    load_ms = (time.perf_counter() - start) * 1000
    texts = list(dict.fromkeys([row['text'] for row in inputs['documents']] + [row['text'] for row in inputs['queries']]))
    embeddings = []
    for text in texts:
        encoded = tokenizer(text, return_tensors='pt', truncation=False)
        if encoded.input_ids.shape[1] > policy['embeddingMaxTokens']:
            raise ValueError('Embedding input exceeds explicit token contract; no truncation permitted')
        began = time.perf_counter()
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            mask = encoded.attention_mask.unsqueeze(-1)
            vector = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), p=2, dim=1)[0]
        if not torch.isfinite(vector).all():
            raise ValueError('Non-finite source embedding')
        embeddings.append({'text': text, 'vector': vector.tolist(), 'tokenIds': encoded.input_ids[0].tolist(),
                           'elapsedMs': (time.perf_counter() - began) * 1000})
    reranking = []
    for query in inputs['queries']:
        scores = []
        for document in inputs['documents']:
            encoded = rank_tokenizer(query['text'], document['text'], return_tensors='pt', truncation=False)
            if encoded.input_ids.shape[1] > policy['rerankerMaxTokens']:
                raise ValueError('Reranker pair exceeds explicit token contract; no truncation permitted')
            began = time.perf_counter()
            with torch.inference_mode():
                value = rank_model(**encoded).logits.reshape(-1)
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError('Expected one finite raw relevance logit')
            scores.append({'text': document['text'], 'score': value.item(),
                           'elapsedMs': (time.perf_counter() - began) * 1000})
        reranking.append({'query': query['text'], 'scores': scores})
        print('Captured', query['id'], flush=True)
    return {'schema': 'doppler.compact-source-capture/v1', 'policy': policy,
            'preparedSha256': digest(inputs_path), 'sources': sources, 'dimension': len(embeddings[0]['vector']),
            'embeddings': embeddings, 'reranking': reranking, 'loadMs': load_ms,
            'torchVersion': torch.__version__, 'transformersVersion': transformers.__version__,
            'surface': 'cpu-float32-source-reference', 'physicalDopplerExecution': False,
            'probeSha256': digest(Path(__file__))}


if __name__ == '__main__':
    policy = json.loads(Path(sys.argv[1]).read_text())
    result = capture(policy)
    with Path(policy['outputPath']).open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
