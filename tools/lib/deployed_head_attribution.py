"""Offline same-hidden-state output-head attribution at a retained token boundary."""
import hashlib
import json
import numpy as np
import torch
from torch.nn.functional import linear
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from .source_task_quality import StoredHalfCache


def probabilities(logits, history, settings, pad):
    assert settings['repetitionPenaltyWindow'] == 0 and settings['presencePenalty'] == 0
    assert settings['stopSequences'] == [] and settings['minP'] == 0
    values = logits.numpy().copy()
    assert np.isfinite(values).all()
    for token in set(history):
        value = float(values[token])
        values[token] = value / settings['repetitionPenalty'] if value > 0 else value * settings['repetitionPenalty']
    if pad is not None:
        values[pad] = -np.inf
    scores, context = torch.from_numpy(values).unsqueeze(0), torch.tensor([history])
    for processor in [TemperatureLogitsWarper(settings['temperature']), TopKLogitsWarper(settings['topK']),
                      TopPLogitsWarper(settings['topP'])]:
        scores = processor(context, scores)
    return torch.softmax(scores, dim=-1)


def probe_deployed_head(model, original_head, config, tokenizer, baseline, task, report, output):
    assert baseline['settings'] == report['settings'] and baseline['source'] == report['source']
    assert baseline['deployed'] == report['deployed']
    row = next(row for row in baseline['results'] if row['id'] == task['caseId'] and row['seed'] == task['seed'])
    step_limit = task['selectedStep']
    assert row['tokenIds'][step_limit] == task['selectedTokenId']
    assert tokenizer.encode(row['renderedPrompt'], add_special_tokens=False).ids == row['inputIds']
    cache = StoredHalfCache(config=config)
    history, current = list(row['inputIds']), row['inputIds']
    random = torch.Generator(device='cpu').manual_seed(row['seed'])
    captured = []
    handle = model.get_output_embeddings().register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone()))
    try:
        for step in range(step_limit + 1):
            captured.clear()
            with torch.inference_mode():
                result = model(input_ids=torch.tensor([current]), past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
            logits = result.logits[0, -1].float()
            assert len(captured) == 1
            hidden = captured[0][0, -1]
            distribution = probabilities(logits, history, report['settings'], config.pad_token_id)
            before = random.get_state().clone()
            chosen = int(torch.multinomial(distribution, 1, generator=random).item())
            assert chosen == row['tokenIds'][step], f'Retained source replay diverged at token {step}'
            if step == step_limit:
                with torch.inference_mode():
                    q4 = linear(captured[0], model.get_output_embeddings().weight)[0, -1]
                    source = linear(captured[0], original_head)[0, -1]
                assert torch.equal(q4, logits), 'Observed head input must exactly reproduce actual deployed logits.'
                source_prob = probabilities(source, history, report['settings'], config.pad_token_id)
                clone = torch.Generator(device='cpu'); clone.set_state(before)
                alternate = int(torch.multinomial(source_prob, 1, generator=clone).item())
                def selected(values, probs, chosen_token):
                    indices = set(torch.topk(probs[0], min(10, probs.shape[-1])).indices.tolist())
                    indices.update([chosen, alternate, *task['eosTokenIds']])
                    return {'chosenToken': chosen_token,
                            'chosenText': tokenizer.decode([chosen_token], skip_special_tokens=False),
                            'tokens': [{'id': index, 'text': tokenizer.decode([index], skip_special_tokens=False),
                                        'logit': float(values[index]), 'probability': float(probs[0, index])}
                                       for index in sorted(indices, key=lambda index: -float(probs[0, index]))]}
                report.update({'schema': 'doppler.deployed-head-attribution/v1', 'caseId': row['id'], 'seed': row['seed'],
                               'selectedStep': step, 'allRetainedPrefixTokensMatch': True,
                               'identicalHiddenOperands': True, 'deployedHeadReplayExact': True,
                               'hiddenSha256': hashlib.sha256(hidden.numpy().tobytes()).hexdigest(),
                               'rngStateSha256': hashlib.sha256(before.numpy().tobytes()).hexdigest(),
                               'sameRngPosition': True, 'selectedTokenChanged': alternate != chosen,
                               'sourceHeadSelectsEos': alternate in task['eosTokenIds'],
                               'deployedHead': selected(q4, distribution, chosen),
                               'originalHead': selected(source, source_prob, alternate),
                               'scope': 'One counterfactual output projection on identical deployed-model hidden state; not quality or numeric qualification.'})
                output.write_text(json.dumps(report) + '\n')
                print(json.dumps({key: report[key] for key in ['selectedStep', 'allRetainedPrefixTokensMatch',
                                  'selectedTokenChanged', 'sourceHeadSelectsEos']}), flush=True)
            history.append(chosen); current = [chosen]
            if step % 64 == 0:
                print(json.dumps({'replayedTokens': step + 1}), flush=True)
    finally:
        handle.remove()
