"""Frozen W8A8 approximation controls using independent source model equations."""
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from lib.source_task_quality import StoredHalfCache


def evaluate_deployed_quality(model, config, manifest, args, report):
    suite_path = Path(args.quality_suite)
    suite = json.loads(suite_path.read_text())
    assert suite['schema'] == 'doppler.w8a8-qualification/v1'
    settings = suite['generation']
    assert settings['temperature'] == 0 and settings['topK'] == 1 and settings['topP'] == 1
    assert settings['repetitionPenalty'] == 1 and settings['presencePenalty'] == 0
    assert settings['suppressTokens'] == [] and settings['stopSequences'] == []
    assert settings['useChatTemplate'] is True and args.source_tokenizer
    source_root = Path(args.source_tokenizer)
    source_tokenizer = AutoTokenizer.from_pretrained(source_root, local_files_only=True, trust_remote_code=False)
    tokenizer = Tokenizer.from_file(str(Path(args.model) / 'tokenizer.json'))
    eos = manifest['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    assert eos == json.loads((source_root / 'generation_config.json').read_text())['eos_token_id']
    source_files = ['config.json', 'generation_config.json', 'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json']
    report.update({'schema': 'doppler.w8a8-source-quality/v1', 'qualified': False,
                   'scope': 'Independent source equations on exact deployed weights; approximation baseline, not GPU acceptance',
                   'suiteSha256': hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                   'qualityToolSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'sourceTokenizerFiles': {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in source_files},
                   'settings': settings, 'eosTokenIds': eos, 'tasks': [], 'teacherForced': []})
    out = Path(args.out)
    assert not out.exists(), 'Preserve prior qualification outputs'

    def save():
        out.write_text(json.dumps(report, indent=2) + '\n')

    try:
        for task in suite['quality']['tasks']:
            prompt = source_tokenizer.apply_chat_template(task['messages'], tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            assert ids == source_tokenizer.encode(prompt, add_special_tokens=False)
            assert len(ids) + settings['maxTokens'] <= settings['maxSeqLen']
            row = {'id': task['id'], 'messages': task['messages'], 'renderedPrompt': prompt,
                   'inputIds': ids, 'tokenIds': [], 'stopReason': None}
            report['tasks'].append(row)
            cache, current = StoredHalfCache(config=config), ids
            for step in range(settings['maxTokens']):
                with torch.inference_mode():
                    result = model(input_ids=torch.tensor([current]), past_key_values=cache,
                                   use_cache=True, logits_to_keep=1)
                    logits = result.logits[0, -1].float()
                    assert torch.isfinite(logits).all(), 'Nonfinite generation logits'
                    token = int(logits.argmax())
                row['tokenIds'].append(token)
                current = [token]
                if token in eos:
                    row['stopReason'] = 'eos'
                    break
                if step + 1 == settings['maxTokens']:
                    row['stopReason'] = 'max-tokens'
            row['text'] = tokenizer.decode(row['tokenIds'], skip_special_tokens=True)
            row['passed'] = row['stopReason'] == 'eos' and row['text'].strip(' \t\r\n\f\v') == task['expected']
            save()
            print(json.dumps({'task': row['id'], 'text': row['text'], 'passed': row['passed'],
                              'stopReason': row['stopReason']}), flush=True)

        for index, document in enumerate(suite['teacherForced']['documents']):
            ids = source_tokenizer.encode(document, add_special_tokens=True)[:suite['teacherForced']['maxTokensPerDocument']]
            assert ids == tokenizer.encode(document, add_special_tokens=True).ids[:len(ids)]
            assert len(ids) >= suite['teacherForced']['minimumTokensPerDocument']
            cache = StoredHalfCache(config=config)
            with torch.inference_mode():
                result = model(input_ids=torch.tensor([ids[:-1]]), past_key_values=cache,
                               use_cache=True, logits_to_keep=0)
                logits = result.logits[0].float()
                assert torch.isfinite(logits).all(), 'Nonfinite teacher-forced logits'
                targets = torch.tensor(ids[1:])
                selected = logits.gather(1, targets[:, None]).squeeze(1).double()
                nll = torch.logsumexp(logits.double(), dim=-1) - selected
                row = {'document': index, 'inputIds': ids, 'targetNll': nll.tolist(),
                       'targetLogits': selected.tolist(), 'top1': logits.argmax(dim=-1).tolist(),
                       'meanNll': float(nll.mean()), 'targetTokenCount': len(ids) - 1}
            report['teacherForced'].append(row)
            save()
            print(json.dumps({'document': index, 'meanNll': row['meanNll']}), flush=True)
        nll_values = [value for row in report['teacherForced'] for value in row['targetNll']]
        report['summary'] = {'correctTasks': sum(row['passed'] for row in report['tasks']),
                             'taskCount': len(report['tasks']), 'meanNll': sum(nll_values) / len(nll_values),
                             'targetTokenCount': len(nll_values), 'approximationAcceptance': 'requires all compared lanes'}
        save()
        return report
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        save()
        raise
