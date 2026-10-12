// Causal diagnostic only: changed activations cannot qualify an implementation.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
const [parentPath, controlPath, interventionPath, output] = process.argv.slice(2);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const read = async path => {
  const bytes = await readFile(path);
  return JSON.parse(bytes[0] === 0x1f && bytes[1] === 0x8b ? gunzipSync(bytes) : bytes);
};
const root = 'artifacts/whole-model-f64-complete-20261011/';
const proposal = await read(root + 'proposal.json');
assert.equal(hash(await readFile('tools/qwen35-f64-reference.py')), proposal.toolSha256);
const index = await read(root + 'independent-output-index.json');
const targets = new Map();
for (const shard of index.shards.filter(shard => shard.index === 2)) {
  assert.equal(hash(await readFile(shard.path)), shard.sha256);
  const record = await read(shard.path);
  assert.equal(record.toolSha256, proposal.toolSha256);
  assert.equal(record.contractSha256, proposal.contracts[2].sha256);
  for (const row of record.results) if (row.step <= 7) targets.set(row.step, Buffer.from(row.logitsF64, 'base64'));
}
assert.equal(targets.size, 8);
const parent = await read(parentPath), control = await read(controlPath), intervention = await read(interventionPath);
assert.equal(parent.archiveSha256, 'c596ecab6d498b3bb6e4d070e65a640713e693020e77c897271481f197875aae');
for (const capture of [control, intervention]) {
  assert(capture.prefixesOnly && capture.executionCompleted && capture.physicalExecutionCompleted);
  assert.deepEqual(capture.declaredShaderMismatches, []);
  assert.equal(capture.results.length, 8);
}
const contract = await read(proposal.contracts[2].path);
const rows = [];
for (let step = 0; step < 8; step++) {
  const original = parent.results.find(row => row.index === 2 && row.step === step);
  const copied = control.results.find(row => row.index === 2 && row.step === step);
  const changed = intervention.results.find(row => row.index === 2 && row.step === step);
  assert.equal(copied.logits, original.logits, 'Copy-only intervention must reproduce every downstream output bit');
  for (const row of [copied, changed]) assert.deepEqual(row.inputIds, contract.inputIds.slice(0, contract.prefixLengths[0]));
  const expected = targets.get(step), baseline = Buffer.from(original.logits, 'base64'), actual = Buffer.from(changed.logits, 'base64');
  assert.equal(expected.length, baseline.length * 2); assert.equal(actual.length, baseline.length);
  const stats = values => {
    let maximum = 0, squared = 0, overTolerance = 0, worstVocabulary = 0;
    for (let i = 0; i < values.length / 4; i++) {
      const value = values.readFloatLE(i * 4), target = expected.readDoubleLE(i * 8);
      assert(Number.isFinite(value) && Number.isFinite(target));
      const error = Math.abs(value - target); squared += error * error;
      if (error > maximum) { maximum = error; worstVocabulary = i; }
      if (error > 0.001) overTolerance++;
    }
    return { maximum, rms: Math.sqrt(squared / (values.length / 4)), overTolerance, worstVocabulary };
  };
  rows.push({ step, parent: stats(baseline), correctedLocalPrefillAttention: stats(actual),
    changedLogits: Array.from({ length: actual.length / 4 }, (_, i) => actual.readUInt32LE(i * 4) !== baseline.readUInt32LE(i * 4)).filter(Boolean).length });
}
const result = { scope: 'Only prefill layer15 independent same-input attention replaced, real GPU downstream; not qualification',
  qualified: false, copyControlExact: true, parentSha256: hash(await readFile(parentPath)),
  controlSha256: hash(await readFile(controlPath)), interventionSha256: hash(await readFile(interventionPath)),
  approvalSha256: hash(await readFile(root + 'approval.json')), rows };
await writeFile(output, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result, null, 2));
