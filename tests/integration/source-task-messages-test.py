"""Ordered source-evaluator history and pending factual-review controls."""
import copy
import json
from pathlib import Path
import runpy
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from lib.source_task_quality import task_messages, compare_task_answer, write_quality_report

original = runpy.run_path('tools/qwen35-original-weight-generation.py')
validate_identity = original['validate_source_identity']
revision = '0123456789abcdef' * 2 + '01234567'
for model_id in ['Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-2B']:
    receipt = {'repository': model_id, 'revision': revision}
    manifest = {'artifactIdentity': {'sourceCheckpointId': model_id}}
    validate_identity(receipt, revision, manifest)
    for invalid_receipt, invalid_revision, invalid_manifest in [
        ({**receipt, 'repository': 'Qwen/Unreviewed'}, revision, manifest),
        (receipt, 'f' * 40, manifest),
        ({**receipt, 'revision': 'z' * 40}, 'z' * 40, manifest),
        (receipt, revision, {'artifactIdentity': {'sourceCheckpointId': 'another-model'}}),
    ]:
        try:
            validate_identity(invalid_receipt, invalid_revision, invalid_manifest)
        except AssertionError:
            pass
        else:
            raise AssertionError('Original source identity mismatch accepted')

with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / 'result.json'
    write_quality_report(output, {'completed': 1})
    original_write = Path.write_text

    def failed_write(path, text):
        original_write(path, text[:5])
        raise OSError('simulated interrupted report write')

    with patch.object(Path, 'write_text', failed_write):
        try:
            write_quality_report(output, {'completed': 2})
        except OSError:
            pass
        else:
            raise AssertionError('Write failure was swallowed')
    assert json.loads(output.read_text()) == {'completed': 1}
    assert not output.with_name('result.json.tmp').exists()
    write_quality_report(output, {'completed': 2})
    assert json.loads(output.read_text()) == {'completed': 2}

history = [{'role': 'user', 'content': 'source [D1:P1]'},
           {'role': 'assistant', 'content': 'retained wrong answer'},
           {'role': 'user', 'content': 'check the original'}]
task = {'messages': history}
snapshot = copy.deepcopy(task)
assert task_messages(task, {}) == history
assert task == snapshot
assert task_messages(task, {'instruction': 'policy'}) == [{'role': 'system', 'content': 'policy'}] + history
assert task_messages({'prompt': 'single'}, {}) == [{'role': 'user', 'content': 'single'}]
for invalid in [{}, {'messages': []}, {'messages': history, 'prompt': 'ambiguous'},
                {'messages': history[:-1]}, {'messages': [{'role': 'system', 'content': 'x'}]},
                {'messages': [{'role': 'user', 'content': ''}]},
                {'messages': [{'role': 'user', 'content': 'x', 'extra': True}]}]:
    try:
        task_messages(invalid, {})
    except AssertionError:
        pass
    else:
        raise AssertionError('Invalid history accepted')

manual = {'reviewCriteria': ['Cite original facts, not retained mistakes.']}
assert compare_task_answer('plausible but unreviewed', manual) is None
try:
    compare_task_answer('A', {**manual, 'expected': 'A'})
except AssertionError:
    pass
else:
    raise AssertionError('Ambiguous review accepted')

# The synthetic execution also checks unchanged EOS/budget/reasoning failures.
controls = runpy.run_path('artifacts/qwen3-17b-application-policy-20261011/evaluator-boundary-check.py')
suite = copy.deepcopy(controls['base'])
suite['cases'] = [{'id': 'ordered-exact', **task, 'expected': 'A'}]
row = controls['check']([2, 4, 3, 4, 5], suite)
assert row['passed'] and row['messages'] == history
suite['cases'] = [{'id': 'ordered-review', **task, **manual}]
row = controls['check']([2, 4, 3, 4, 5], suite)
assert row['messages'] == history and row['completionPassed']
assert row['reviewStatus'] == 'pending' and row['passed'] is None
row = controls['check']([2, 4, 5], suite)
assert not row['completionPassed'] and row['passed'] is False
print('source-task-messages-test: pass')
