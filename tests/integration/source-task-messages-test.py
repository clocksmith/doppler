"""Ordered source-evaluator history and pending factual-review controls."""
import copy
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from lib.source_task_quality import task_messages, compare_task_answer

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
suite['cases'] = [{'id': 'ordered-review', **task, **manual}]
row = controls['check']([2, 4, 3, 4, 5], suite)
assert row['messages'] == history and row['completionPassed']
assert row['reviewStatus'] == 'pending' and row['passed'] is None
row = controls['check']([2, 4, 5], suite)
assert not row['completionPassed'] and row['passed'] is False
print('source-task-messages-test: pass')
