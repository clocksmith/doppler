"""Offline source-model task comparison shared by retained quality controls."""
import json
import numpy as np
import torch
from transformers import DynamicCache


class StoredHalfCache(DynamicCache):
    """Round stored K/V to the declared F16 policy; source math remains F32."""
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        keys, values = super().update(key_states.half(), value_states.half(), layer_idx, *args, **kwargs)
        return keys.float(), values.float()



def evaluate_tasks(model, config, tokenizer, source_tokenizer, suite, report, output):
    settings = suite['generation']
    assert settings['temperature'] == 0 and settings['topK'] == 1 and settings['topP'] == 1
    assert settings['repetitionPenaltyWindow'] == 0 and settings['presencePenalty'] == 0
    assert settings['stopSequences'] == []
    for task in suite['cases']:
        template_options = {'tokenize': False, 'add_generation_prompt': True}
        if 'enableThinking' in suite:
            template_options['enable_thinking'] = suite['enableThinking']
        rendered = source_tokenizer.apply_chat_template([{'role': 'user', 'content': task['prompt']}],
                                                        **template_options)
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        assert ids == source_tokenizer.encode(rendered, add_special_tokens=False)
        assert len(ids) + settings['maxTokens'] <= settings['maxSeqLen']
        row = {'id': task['id'], 'renderedPrompt': rendered, 'inputIds': ids,
               'tokenIds': [], 'stopReason': None}
        report['results'].append(row)
        cache, history, current = StoredHalfCache(config=config), list(ids), ids
        for step in range(settings['maxTokens']):
            with torch.inference_mode():
                result = model(input_ids=torch.tensor([current]), past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
                logits = result.logits[0, -1].float().numpy().copy()
            assert np.isfinite(logits).all()
            raw_logits = logits.copy() if step == 0 else None
            for token in set(history):
                value = float(logits[token])
                logits[token] = value / settings['repetitionPenalty'] if value > 0 else value * settings['repetitionPenalty']
            if config.pad_token_id is not None:
                logits[config.pad_token_id] = -np.inf
            if step == 0:
                def top(values):
                    return [{'token': int(index), 'text': tokenizer.decode([int(index)], skip_special_tokens=False),
                             'logit': float(values[index]), 'presentInPrompt': int(index) in history}
                            for index in np.argsort(values)[-10:][::-1]]
                row['firstStep'] = {'raw': top(raw_logits), 'afterRepetitionPenalty': top(logits)}
            token = int(logits.argmax())
            row['tokenIds'].append(token)
            history.append(token)
            current = [token]
            if token == config.eos_token_id:
                row['stopReason'] = 'eos'
            elif step + 1 == settings['maxTokens']:
                row['stopReason'] = 'max-tokens'
            if step % 64 == 0 or row['stopReason']:
                row['text'] = tokenizer.decode(row['tokenIds'], skip_special_tokens=True)
                output.write_text(json.dumps(report) + '\n')
                print(json.dumps({'id': row['id'], 'step': step, 'stopReason': row['stopReason'],
                                  'textTail': row['text'][-160:]}), flush=True)
            if row['stopReason']:
                break
        row['passed'] = row['stopReason'] == 'eos' and row['text'].split() == task['expected'].split()
        output.write_text(json.dumps(report) + '\n')
    report['passed'] = all(row['passed'] for row in report['results'])
    output.write_text(json.dumps(report) + '\n')
