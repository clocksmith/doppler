"""Prospective cited-value comparison; historical strict task contracts stay unchanged."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from lib.source_task_quality import compare_task_answer


def task(value, citation):
    return {'expected': value + ' ' + citation,
            'comparison': {'schema': 'doppler.cited-value-comparison/v1', 'value': value, 'citation': citation}}


amount = task('$81.20', '[P7]')
for answer in ['$81.20[P7]', '$81.20 [P7]', '$81.20\t[P7]', '\n $81.20 \t [P7]\n']:
    assert compare_task_answer(answer, amount), answer
for answer in ['$81.21[P7]', '$81.20[P8]', '$81.20 [ P7]', '$81.20[P7].',
               'Total: $81.20[P7]', '$81.20[P7] extra', '$81.20\n[P7]',
               '$ 81.20[P7]', '$81. 20[P7]', '$81.20\u00a0[P7]', '$81.20[P7][P7]']:
    assert not compare_task_answer(answer, amount), answer
choice = task('Yes', '[P2]')
assert compare_task_answer('Yes[P2]', choice)
for answer in ['No[P2]', 'yes[P2]', 'Y es[P2]', 'Yes.[P2]', 'Yes[P2].']:
    assert not compare_task_answer(answer, choice), answer

# Original tasks retain whitespace-token comparison, including punctuation rejection.
assert compare_task_answer('\n No \n', {'expected': 'No'})
assert not compare_task_answer('No.', {'expected': 'No'})
assert not compare_task_answer('$81.20[P7]', {'expected': '$81.20 [P7]'})
assert compare_task_answer('1\t2\n3', {'expected': '1\n2\n3'})
assert not compare_task_answer('123', {'expected': '1\n2\n3'})

invalid = [
    {**amount, 'comparison': None},
    {**amount, 'comparison': {**amount['comparison'], 'schema': 'unknown'}},
    {**amount, 'comparison': {**amount['comparison'], 'ignorePunctuation': True}},
    task('$ 81.20', '[P7]'), task('$81.20', 'P7'), task('$81.20', '[P 7]'),
    {**amount, 'expected': '$81.21 [P7]'},
]
for invalid_task in invalid:
    try:
        compare_task_answer(invalid_task['expected'], invalid_task)
    except AssertionError:
        pass
    else:
        raise AssertionError('Invalid comparison contract accepted.')
print('source-task-answer-contract-test: pass')
