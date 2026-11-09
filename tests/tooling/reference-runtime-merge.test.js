import assert from 'node:assert/strict';
import fs from 'node:fs';

import { withReferenceTranscriptRuntimeConfig, normalizeRuntimeConfigInput } from '../../src/tooling/reference-verify.js';
import { mergeRuntimeValues } from '../../src/config/runtime-merge.js';

// The reference-transcript patch merges with the declared runtime semantics:
// sibling keys survive, arrays replace whole, null resets, undefined keeps.
const merged = withReferenceTranscriptRuntimeConfig({
  runtimeConfig: { shared: { harness: { referenceTranscript: { captureLogits: false, sampleEvery: 4 }, other: { keep: true } }, pins: [1, 2] } },
  modelUrl: 'file:///x',
});
assert.equal(merged.modelUrl, 'file:///x');
assert.deepEqual(merged.runtimeConfig.shared.harness.referenceTranscript, { enabled: true, captureLogits: true, captureKvBytes: true, sampleEvery: 4 });
assert.deepEqual(merged.runtimeConfig.shared.harness.other, { keep: true });
assert.deepEqual(merged.runtimeConfig.shared.pins, [1, 2]);

// The copies the review found erased a base value under a nested undefined
// override; the canonical helper keeps it.
assert.deepEqual(mergeRuntimeValues({ harness: { keep: 1, drop: 2 } }, { harness: { keep: undefined, drop: null } }), { harness: { keep: 1, drop: null } });

assert.deepEqual(await normalizeRuntimeConfigInput({ a: 1 }), { runtimeConfig: { a: 1 } });
assert.deepEqual(await normalizeRuntimeConfigInput('{"b":2}'), { runtimeConfig: { b: 2 } });
assert.deepEqual(await normalizeRuntimeConfigInput(''), {});

// Neither runtime-merge consumer keeps a private merge routine any more.
for (const file of ['src/tooling/reference-verify.js', 'tools/run-program-bundle-reference.js']) {
  const source = fs.readFileSync(new URL(`../../${file}`, import.meta.url), 'utf8');
  assert.ok(!/function mergePlainObjects/.test(source), `${file} defines no merge copy`);
  assert.ok(/mergeRuntimeValues/.test(source), `${file} uses the canonical runtime merge`);
}
assert.ok(!/async function normalizeRuntimeConfigInput/.test(fs.readFileSync(new URL('../../tools/run-program-bundle-reference.js', import.meta.url), 'utf8')), 'the CLI imports the shared runtime-config normalizer');

console.log('reference-runtime-merge: canonical merge semantics reach the reference verifier and CLI');
