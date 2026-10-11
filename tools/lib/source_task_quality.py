"""Offline source-model task comparison shared by retained quality controls."""
import json
import re
import numpy as np
import torch
from transformers import DynamicCache
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper


def write_quality_report(output, report):
    """Publish a complete snapshot so concurrent observers cannot copy torn JSON."""
    temporary = output.with_name(output.name + '.tmp')
    try:
        temporary.write_text(json.dumps(report) + '\n')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def task_messages(task, suite):
    """Keep frozen user-ended history intact; never flatten or invent a turn."""
    assert ('prompt' in task) != ('messages' in task), 'Declare prompt or messages, not both.'
    messages = []
    if 'instruction' in suite:
        assert isinstance(suite['instruction'], str) and suite['instruction'].strip()
        messages.append({'role': 'system', 'content': suite['instruction']})
    if 'prompt' in task:
        assert isinstance(task['prompt'], str) and task['prompt'].strip()
        return messages + [{'role': 'user', 'content': task['prompt']}]
    history = task['messages']
    assert isinstance(history, list) and history and len(history) % 2 == 1
    for index, message in enumerate(history):
        assert isinstance(message, dict) and set(message) == {'role', 'content'}
        assert message['role'] == ('user' if index % 2 == 0 else 'assistant')
        assert isinstance(message['content'], str) and message['content'].strip()
        messages.append(dict(message))
    return messages


def compare_task_answer(answer, task):
    """Apply only the frozen task's declared output contract, never rewrite output."""
    if 'reviewCriteria' in task:
        assert 'expected' not in task and 'comparison' not in task
        criteria = task['reviewCriteria']
        assert isinstance(criteria, list) and criteria
        assert all(isinstance(item, str) and item.strip() for item in criteria)
        return None  # Free-form factual review cannot be inferred from EOS or keywords.
    if 'comparison' not in task:
        return answer.split() == task['expected'].split()
    contract = task['comparison']
    assert isinstance(contract, dict) and set(contract) == {'schema', 'value', 'citation'}
    assert contract['schema'] == 'doppler.cited-value-comparison/v1'
    value, citation = contract['value'], contract['citation']
    assert isinstance(value, str) and value and not any(character.isspace() for character in value)
    assert isinstance(citation, str) and re.fullmatch(r'\[[A-Za-z0-9_-]+\]', citation)
    assert task['expected'] == value + ' ' + citation
    return re.fullmatch(re.escape(value) + r'[ \t]*' + re.escape(citation), answer.strip()) is not None


class StoredHalfCache(DynamicCache):
    """Round stored K/V to the declared F16 policy; source math remains F32."""
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        keys, values = super().update(key_states.half(), value_states.half(), layer_idx, *args, **kwargs)
        return keys.float(), values.float()


def probe_retained_prefix(model, config, tokenizer, baseline, case_id, selected_step, report, output):
    """Observe policy at one retained boundary while requiring exact prefix replay."""
    row = next(row for row in baseline['results'] if row['id'] == case_id)
    assert 0 <= selected_step < len(row['tokenIds'])
    assert baseline['settings'] == report['settings']
    assert baseline['settings']['temperature'] == 0
    assert baseline['source'] == report['source'] and baseline['suiteSha256'] == report['suiteSha256']
    ids = row['inputIds']
    assert tokenizer.encode(row['renderedPrompt'], add_special_tokens=False).ids == ids
    cache, history, current = StoredHalfCache(config=config), list(ids), ids
    for step in range(selected_step + 1):
        with torch.inference_mode():
            result = model(input_ids=torch.tensor([current]), past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            raw = result.logits[0, -1].float().numpy().copy()
        assert np.isfinite(raw).all()
        adjusted = raw.copy()
        for token in set(history):
            value = float(adjusted[token])
            penalty = report['settings']['repetitionPenalty']
            adjusted[token] = value / penalty if value > 0 else value * penalty
        if config.pad_token_id is not None:
            adjusted[config.pad_token_id] = -np.inf
        winner = int(adjusted.argmax())
        assert winner == row['tokenIds'][step], f'Retained prefix diverged at {step}'
        if step == selected_step:
            def top(values):
                return [{'token': int(index), 'text': tokenizer.decode([int(index)], skip_special_tokens=False),
                         'logit': float(values[index]), 'seenInHistory': int(index) in history}
                        for index in np.argsort(values)[-10:][::-1]]
            report.update({'schema': 'doppler.source-policy-boundary/v1', 'caseId': case_id,
                           'selectedStep': selected_step, 'allRetainedPrefixTokensMatch': True,
                           'inputIds': ids, 'generatedPrefix': row['tokenIds'][:step],
                           'raw': top(raw), 'afterRepetitionPenalty': top(adjusted),
                           'winnerChangedByPenalty': int(raw.argmax()) != winner})
            write_quality_report(output, report)
            print(json.dumps({'caseId': case_id, 'step': step, 'raw': report['raw'][:3],
                              'adjusted': report['afterRepetitionPenalty'][:3]}), flush=True)
        history.append(winner)
        current = [winner]



def evaluate_tasks(model, config, tokenizer, source_tokenizer, suite, report, output):
    for task in suite['cases']:
        task_messages(task, suite)
        if 'reviewCriteria' in task:
            assert compare_task_answer('', task) is None
        else:
            assert compare_task_answer(task['expected'], task), 'Invalid frozen answer contract.'
    settings = suite['generation']
    greedy = settings['temperature'] == 0
    if greedy:
        assert settings['topK'] == 1 and settings['topP'] == 1
    else:
        assert settings['temperature'] > 0 and settings['topK'] > 0 and 0 < settings['topP'] <= 1
        assert settings['minP'] == 0 and len(set(suite['seeds'])) == len(suite['seeds'])
    assert settings['repetitionPenaltyWindow'] == 0 and settings['presencePenalty'] == 0
    assert settings['stopSequences'] == []
    eos_tokens = suite.get('eosTokenIds', [config.eos_token_id])
    if 'answerBoundary' in suite:
        assert suite['enableThinking'] is True and suite['answerBoundary']['exactlyOne'] is True
        assert tokenizer.encode(suite['answerBoundary']['text'], add_special_tokens=False).ids == [suite['answerBoundary']['tokenId']]
    for task, seed in ((task, seed) for seed in suite.get('seeds', [None]) for task in suite['cases']):
        random = None if greedy else torch.Generator(device='cpu').manual_seed(seed)
        template_options = {'tokenize': False, 'add_generation_prompt': True}
        if 'enableThinking' in suite:
            template_options['enable_thinking'] = suite['enableThinking']
        messages = task_messages(task, suite)
        rendered = source_tokenizer.apply_chat_template(messages, **template_options)
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        assert ids == source_tokenizer.encode(rendered, add_special_tokens=False)
        assert len(ids) + settings['maxTokens'] <= settings['maxSeqLen']
        row = {'id': task['id'], 'messages': messages, 'renderedPrompt': rendered, 'inputIds': ids,
               'tokenIds': [], 'stopReason': None}
        if seed is not None:
            row['seed'] = seed
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
            if greedy:
                token = int(logits.argmax())
            else:
                scores, context = torch.from_numpy(logits).unsqueeze(0), torch.tensor([history])
                for processor in [TemperatureLogitsWarper(settings['temperature']),
                                  TopKLogitsWarper(settings['topK']), TopPLogitsWarper(settings['topP'])]:
                    scores = processor(context, scores)
                token = int(torch.multinomial(torch.softmax(scores, dim=-1), 1, generator=random).item())
            row['tokenIds'].append(token)
            history.append(token)
            current = [token]
            if token in eos_tokens:
                row['stopReason'] = 'eos'
            elif step + 1 == settings['maxTokens']:
                row['stopReason'] = 'max-tokens'
            if 'maxOutputCharacters' in suite:
                raw_text = tokenizer.decode(row['tokenIds'], skip_special_tokens=False)
                if len(raw_text.encode('utf-16-le')) // 2 > suite['maxOutputCharacters']:
                    row['stopReason'] = 'output-character-limit'
            if step % 64 == 0 or row['stopReason']:
                row['text'] = tokenizer.decode(row['tokenIds'], skip_special_tokens=True)
                write_quality_report(output, report)
                print(json.dumps({'id': row['id'], 'step': step, 'stopReason': row['stopReason'],
                                  'textTail': row['text'][-160:]}), flush=True)
            if row['stopReason']:
                break
        answer = row['text']
        boundary_valid = True
        if 'answerBoundary' in suite:
            marker = suite['answerBoundary']['tokenId']
            positions = [index for index, token in enumerate(row['tokenIds']) if token == marker]
            boundary_valid = len(positions) == 1
            answer = tokenizer.decode(row['tokenIds'][positions[0] + 1:], skip_special_tokens=True) if boundary_valid else ''
            row.update({'rawDecodedText': tokenizer.decode(row['tokenIds'], skip_special_tokens=False),
                        'answerText': answer, 'answerBoundaryPositions': positions,
                        'generatedTokenCount': len(row['tokenIds']), 'totalTokenCount': len(ids) + len(row['tokenIds'])})
            row['rawDecodedCharacters'] = len(row['rawDecodedText'].encode('utf-16-le')) // 2
            row['characterUnit'] = 'UTF-16 code units, matching JavaScript string.length'
            row['withinBudgets'] = (row['rawDecodedCharacters'] <= suite['maxOutputCharacters']
                                    and row['generatedTokenCount'] <= settings['maxTokens']
                                    and row['totalTokenCount'] <= settings['maxSeqLen'])
            boundary_valid = boundary_valid and row['withinBudgets']
        completion_passed = row['stopReason'] == 'eos' and boundary_valid and bool(answer.strip())
        answer_passed = compare_task_answer(answer, task)
        row['passed'] = completion_passed and answer_passed
        if 'reviewCriteria' in task:
            row.update({'completionPassed': completion_passed,
                        'reviewCriteria': task['reviewCriteria'], 'reviewStatus': 'pending'})
        write_quality_report(output, report)
    report['passed'] = bool(report['results']) and all(row['passed'] is True for row in report['results'])
    report['reviewRequired'] = any(row.get('reviewStatus') == 'pending' for row in report['results'])
    write_quality_report(output, report)
