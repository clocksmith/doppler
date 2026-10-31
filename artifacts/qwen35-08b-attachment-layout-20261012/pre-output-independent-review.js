// Independent frozen-input audit. This script never reads candidate outputs.
import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';

const root = new URL('./', import.meta.url);
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const read = name => JSON.parse(fs.readFileSync(new URL(`${name}.json`, root)));
const expected = {
  'question-policy-after': 'e57545f7db48a285a35f8c0f74fb101e05fee7ba157d16db7e958702697954f7',
  'holdout-before': '1cdd2af36428843938028d3a0cb9aa79dfd3fddea651c31e1afeeeed48b664b8',
};
for (const [name, digest] of Object.entries(expected)) {
  assert.equal(hash(fs.readFileSync(new URL(`${name}.json`, root))), digest);
}
const evaluatorSha256 = hash(fs.readFileSync(new URL('../../tools/lib/source_task_quality.py', root)));
assert.equal(evaluatorSha256, '37586f498f393570e499859deb0bcaba520e86bb85a03365c176a4981079127f');
const candidate = read('question-policy-after');
const oldSuites = [read('question-before'), read('holdout-before')];
const originals = oldSuites.flatMap(suite => suite.cases);
const cases = [];
for (const actual of candidate.cases) {
  const original = originals.find(row => row.id === actual.id);
  assert.ok(original);
  assert.deepEqual(actual.reviewCriteria, original.reviewCriteria);
  assert.equal(actual.messages.length, original.messages.length);
  let relocatedMessages = 0;
  for (let index = 0; index < original.messages.length; index++) {
    const before = original.messages[index], after = actual.messages[index];
    assert.equal(before.role, after.role);
    const start = before.content.indexOf('<reploid-document-sources-v1>');
    if (start < 0) { assert.equal(after.content, before.content); continue; }
    const end = before.content.indexOf('</reploid-document-sources-v1>')
      + '</reploid-document-sources-v1>'.length;
    assert.equal(before.content.slice(end).trim(), '');
    const questionAndPolicy = before.content.slice(0, start).trimEnd();
    const evidence = before.content.slice(start, end);
    assert.equal(after.content, `${evidence}\n\n${questionAndPolicy}`);
    relocatedMessages++;
  }
  assert.equal(relocatedMessages, 1);
  cases.push({ id: actual.id, criteriaUnchanged: true,
    historyUnchangedExceptWholeBlockRelocation: true, relocatedMessages });
}
assert.equal(cases.length, 10);
assert.equal(new Set(cases.map(row => row.id)).size, 10);
for (const original of oldSuites) {
  for (const key of ['generation', 'maxOutputCharacters', 'eosTokenIds', 'enableThinking', 'modelIdentity']) {
    assert.deepEqual(candidate[key], original[key]);
  }
}
const report = {
  schema: 'doppler.pre-output-independent-review/v1',
  reviewer: 'native_participation', candidateOutputsRead: false,
  scriptSha256: hash(fs.readFileSync(new URL(import.meta.url))),
  suites: expected, evaluatorSha256, checksPassed: true, cases,
  semanticReview: {
    disposition: 'Frozen criteria adequately require meaning; no changes requested.',
    criteriaAreConjunctive: true,
    evaluatorAutomaticallyApprovesFreeformAnswers: false,
    requiredManualChecks: [
      'Correct amount must describe the requested quantity or balance, without contradictory totals or invented fees elsewhere.',
      'Preserve negation, unknown status and may-versus-will; quoting a condition without applying it does not answer the question.',
      'Exact citation identifiers must support their attached claims; swapped or unrelated citations do not pass.',
      'Correction turns must resolve retained assistant errors from original passages, not treat earlier assistant statements as authority.',
      'EOS, nonempty text and budget compliance do not establish semantic completion; reject unfinished or incoherent conclusions.',
      'Every criterion in every case remains required; success on selected cases cannot qualify the configuration.',
    ],
    examplesRequiringRejection: [
      'An answer containing $89.50 but adding an unsupported extra fee.',
      'An answer quoting may ship Thursday but concluding shipment is guaranteed.',
      'A $29 answer citing the unit-price passage for the delivery charge and the delivery passage for unit price.',
      'Correct numeric/citation tokens followed by an unfinished or contradictory final conclusion.',
    ],
    boundary: 'Input/rubric/evaluator review only; not candidate answer quality, GPU execution or deployment qualification.',
  },
};
fs.writeFileSync(new URL('pre-output-independent-review.json', root), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ checksPassed: true, cases: cases.length, candidateOutputsRead: false }));
