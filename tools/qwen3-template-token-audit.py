#!/usr/bin/env python3
"""Compare pinned source and existing runtime template tokens without generation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from transformers import AutoTokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    source, suite_path = Path(args.source), Path(args.suite)
    suite = json.loads(suite_path.read_text())
    receipt = json.loads((source / 'download-receipt.json').read_text())
    assert receipt['source'] == suite['modelId'] and receipt['revision'] == suite['sourceRevision']
    for entry in receipt['files']:
        if entry['path'] in ('tokenizer.json', 'tokenizer_config.json', 'config.json'):
            assert digest(source / entry['path']) == entry['sha256']
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    cases = [{'id': task['id'], 'messages': [{'role': 'user', 'content': task['prompt']}]}
             for task in suite['cases']]
    cases += [
        {'id': 'system-and-user', 'messages': [
            {'role': 'system', 'content': 'Answer using the supplied document.'},
            {'role': 'user', 'content': 'Which date is authorized?'}]},
        {'id': 'final-answer-history', 'messages': [
            {'role': 'user', 'content': 'What is two plus three?'},
            {'role': 'assistant', 'content': '5'},
            {'role': 'user', 'content': 'Now add four.'}]},
        {'id': 'raw-reasoning-history', 'messages': [
            {'role': 'user', 'content': 'What is two plus three?'},
            {'role': 'assistant', 'content': '<think>\nI will add.\n</think>\n\n5'},
            {'role': 'user', 'content': 'Now add four.'}]},
        {'id': 'separate-reasoning-history', 'messages': [
            {'role': 'user', 'content': 'What is two plus three?'},
            {'role': 'assistant', 'content': '5', 'reasoning_content': 'I will add.'},
            {'role': 'user', 'content': 'Now add four.'}]},
        {'id': 'assistant-ended-continuation', 'messages': [
            {'role': 'user', 'content': 'What is two plus three?'},
            {'role': 'assistant', 'content': '5'}]},
    ]
    requests = [{**case, 'template': template, 'thinking': thinking}
                for case in cases for template in ['qwen', 'chatml'] for thinking in [True, False]]
    javascript = """
      import fs from 'node:fs';
      import { formatChatMessages } from './src/inference/pipelines/text/chat-format.js';
      import { applyChatTemplate } from './src/inference/pipelines/text/init-chat-templates.js';
      const rows = JSON.parse(fs.readFileSync(0, 'utf8')).map(row => ({
        formatted: formatChatMessages(row.messages, row.template, {thinking:row.thinking}),
        single: row.messages.length === 1 ? applyChatTemplate(row.messages[0].content, row.template,
          {thinking:row.thinking}) : null,
      }));
      process.stdout.write(JSON.stringify(rows));
    """
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['node', '--input-type=module', '--eval', javascript], cwd=root,
                            input=json.dumps(requests), text=True, capture_output=True, check=True)
    rows = []
    for request, actual in zip(requests, json.loads(result.stdout), strict=True):
        expected = tokenizer.apply_chat_template(request['messages'], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=request['thinking'])
        expected_ids = tokenizer.encode(expected, add_special_tokens=False)
        actual_ids = tokenizer.encode(actual['formatted'], add_special_tokens=False)
        first = next((i for i, (a, b) in enumerate(zip(expected_ids, actual_ids)) if a != b),
                     min(len(expected_ids), len(actual_ids)))
        assert actual['single'] is None or actual['single'] == actual['formatted']
        rows.append({**request, 'equal': expected_ids == actual_ids, 'expectedText': expected,
                     'actualText': actual['formatted'], 'expectedTokenIds': expected_ids,
                     'actualTokenIds': actual_ids, 'firstDifference': None if expected_ids == actual_ids else first,
                     'singleAndMessagesAgree': actual['single'] is None or actual['single'] == actual['formatted']})
    ordinary = [row for row in rows if row['id'] not in ('raw-reasoning-history', 'assistant-ended-continuation')]
    for row in ordinary:
        assert row['equal'] == ((row['template'] == 'chatml') == row['thinking'])
        if row['template'] == 'qwen' and row['thinking']:
            assert row['actualTokenIds'] == row['expectedTokenIds'] + tokenizer.encode('<think>\n', add_special_tokens=False)
    report = {'schema': 'doppler.qwen3-template-token-audit/v1', 'generationExecuted': False,
              'ordinaryHistoryComparisonsPassed': len(ordinary),
              'unsupportedHistoryShapes': ['raw-reasoning-history', 'assistant-ended-continuation'],
              'sourceRevision': receipt['revision'], 'suiteSha256': digest(suite_path),
              'producerSha256': digest(__file__),
              'sourceTokenizerConfigSha256': digest(source / 'tokenizer_config.json'),
              'runtimeSources': {name: digest(root / name) for name in [
                  'src/inference/pipelines/text/chat-format.js',
                  'src/inference/pipelines/text/init-chat-templates.js']}, 'results': rows}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{'id': r['id'], 'template': r['template'], 'thinking': r['thinking'],
                       'equal': r['equal'], 'firstDifference': r['firstDifference']} for r in rows]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'suite', 'out']:
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
