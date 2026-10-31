/** Fixed two-prefix public-path control; not full-model qualification. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';

const [capturePath, referencePath, outputPath] = process.argv.slice(2);
assert(capturePath && referencePath && outputPath, 'Expected capture, approved assembled reference, output');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const captureBytes = readFileSync(capturePath);
const capture = JSON.parse(captureBytes[0] === 0x1f ? gunzipSync(captureBytes) : captureBytes);
const referenceBytes = readFileSync(referencePath);
assert.equal(hash(referenceBytes), 'ad26dfbc38c5b890f48da2f1e4313dac9295e39e12f688411a10bf614d22fdd6');
const reference = JSON.parse(referenceBytes);
const contractBytes = readFileSync('artifacts/whole-model-f64-complete-20261011/request-0-contract.json');
assert.equal(hash(contractBytes), '0510eb0aba9ebba1cfd92ec5b8fec0605bb49b03ff1f6a7dc3fc0bdae9e1130e');
const contract = JSON.parse(contractBytes);
assert.equal(capture.archiveSha256, '102fd0c4900c6744fb7d103a4b3609365b57e8538657bd47b8dfd8e7ac3469e7');
assert.equal(capture.manifestIdentity, 'sha256:ef4543d3f909c5c8324bbfe69e8df9665fe3702deabb6b8fbdfca10dcd4fc32e');
assert.equal(capture.pairedGraphSha256, '36b7799edce21590e5846909ddaeea99be05cccaf87b65264d7281b2497fb0f3');
assert.equal(capture.runtimeVersion, '0.6.28');
assert.equal(capture.intermediatePrecision, 'two-f32');
assert.equal(capture.recurrentStatePrecision, 'two-f32');
assert.equal(capture.maxDeviceBytes, 1420000000);
assert.equal(capture.executionCompleted, true);
assert.equal(capture.physicalExecutionCompleted, true);
assert.equal(capture.prefixesOnly, true);
assert.equal(capture.observation, null);
assert.deepEqual(capture.declaredShaderMismatches, []);
assert.deepEqual(capture.results.map(row => [row.index, row.step]), [[0, 0], [0, 1]]);
const rows = capture.results.map(row => {
  assert.deepEqual(row.inputIds, contract.inputIds.slice(0, contract.prefixLengths[0]));
  const target = reference.expected[0].steps[row.step];
  assert.equal(target.logitsDtype, 'f64');
  const actual = Buffer.from(row.logits, 'base64');
  const expected = Buffer.from(target.logits, 'base64');
  assert.equal(actual.length * 2, expected.length);
  assert.equal(actual.length, 248320 * 4);
  let maximum = 0, sumSquares = 0, overTolerance = 0, worstVocabulary = 0;
  for (let index = 0; index < actual.length / 4; index++) {
    const a = actual.readFloatLE(index * 4), b = expected.readDoubleLE(index * 8);
    assert(Number.isFinite(a) && Number.isFinite(b));
    const error = Math.abs(a - b);
    if (error > maximum) { maximum = error; worstVocabulary = index; }
    sumSquares += error * error;
    if (error > 0.001) overTolerance++;
  }
  return { index: 0, step: row.step, values: actual.length / 4, maximum,
    rms: Math.sqrt(sumSquares / (actual.length / 4)), overTolerance, worstVocabulary };
});
const result = { scope: 'Exact 0.6.28 public unsplit first request, prefill and one decode; not all55 or signed qualification',
  captureSha256: hash(captureBytes), approvedReferenceSha256: hash(referenceBytes),
  comparatorSha256: hash(readFileSync(new URL(import.meta.url))), archiveSha256: capture.archiveSha256,
  tolerance: 0.001, rows, passed: rows.every(row => row.overTolerance === 0) };
writeFileSync(outputPath, JSON.stringify(result, null, 2) + '\n');
assert(result.passed, 'Public control exceeds the approved tolerance');
console.log(JSON.stringify(result));
