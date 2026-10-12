// CPU-only review. This does not replace or execute the authoritative evaluator.
import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import zlib from 'node:zlib';
const root = new URL('../../../', import.meta.url);
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const source = fs.readFileSync(new URL('tools/qwen35-f64-reference.py', root), 'utf8');
assert.equal(hash(source), '90376ddec6e6bf997425ee0e1ecf1d98533434ebbe3f934ff00a878dd49e8bc7');
const originalFunction = source.slice(source.indexOf('def delta_history('), source.indexOf('\n\ndef evaluate('));
const boundary = '        state += key[:, :, None] * delta[:, None, :]';
assert.equal(originalFunction.split(boundary).length, 2);
const controlledFunction = originalFunction.replace(boundary,
  `${boundary}\n        state = state.astype(np.float32).astype(np.float64)`);
assert.equal(hash(controlledFunction), 'd5e01474a0b0b5bf6d9a729cee0fe135f040e2ed3a040e4151620f43de38c66e');
assert.ok(controlledFunction.indexOf('state = state.astype') < controlledFunction.indexOf('        output ='));
const shaderHashes = {};
for (const name of ['gated_delta_recurrent', 'gated_delta_fused_decode']) {
  const shader = fs.readFileSync(new URL(`src/gpu/kernels/${name}.wgsl`, root), 'utf8');
  shaderHashes[name] = hash(shader);
  const decay = shader.indexOf('recurrent_state[state_idx] = recurrent_state[state_idx] * g_exp;');
  const update = shader.indexOf('recurrent_state[state_idx] = fma(k_normed, delta, recurrent_state[state_idx]);');
  const output = shader.indexOf('out_value = fma(recurrent_state[state_idx], q_normed, out_value);');
  assert.ok(decay >= 0 && decay < update && update < output);
}
const numpy = JSON.parse(fs.readFileSync(new URL('recurrent-storage-numpy-rounding-review.json', import.meta.url)));
const view = new DataView(new ArrayBuffer(4));
const value = bits => { view.setUint32(0, bits); return view.getFloat32(0); };
const bits = number => { view.setFloat32(0, number); return view.getUint32(0); };
for (const row of numpy.rows) {
  const lower = Number(row.lowerBits), low = value(lower), high = value(lower + 1);
  const input = ((low + high) / 2 + row.ulpOffset * (high - low)) * (row.negative ? -1 : 1);
  assert.equal(bits(input), Number(row.expectedBits));
  assert.equal(Number(row.actualBits), Number(row.expectedBits));
}
assert.equal(numpy.rows.length, 60);
const receiptBytes = zlib.gunzipSync(fs.readFileSync(new URL('../recurrent-materialization-comparison.json.gz', import.meta.url)));
const receipt = JSON.parse(receiptBytes);
assert.equal(receipt.approvedVectorsExact, true);
assert.equal(receipt.evaluatorSha256, hash(source));
const control = receipt.recurrentMaterializationControl;
assert.equal(control.controlledSourceSha256, hash(controlledFunction));
assert.equal(control.perPrefixMaximum.length, 51);
assert.equal(control.perPrefixMaximum.filter(value => value > 0.001).length, 7);
const comparison = control.runtimeComparison;
const bindingBytes = fs.readFileSync(new URL('../recurrent-runtime-binding-validation.json', import.meta.url));
const binding = JSON.parse(bindingBytes);
assert.equal(binding.passed, true);
assert.equal(binding.rows, 51);
assert.equal(binding.runtimeSha256, comparison.sha256);
assert.equal(binding.archiveSha256, comparison.archiveSha256);
assert.equal(binding.auditSha256, hash(fs.readFileSync(new URL('tools/qwen35-short-prefix-audit.py', root))));
const contract = JSON.parse(fs.readFileSync(new URL('artifacts/whole-model-f64-complete-20261011/request-2-contract.json', root)));
assert.deepEqual(binding.inputIds, contract.inputIds);
const report = {
  schema: 'doppler.recurrent-storage-independent-review/v1',
  scriptSha256: hash(fs.readFileSync(new URL(import.meta.url))),
  receiptSha256: hash(receiptBytes), evaluatorSha256: hash(source),
  controlledSourceSha256: hash(controlledFunction), shaderHashes,
  runtimeBinding: { reviewedCorrectionCommit: 'e148f2ed', passed: true,
    receiptSha256: hash(bindingBytes), exactForcedInputMatchesContract: true },
  roundingControls: { cases: 60, numpy: numpy.numpyVersion, independentJavaScriptBitsAgree: true },
  baseline: 'Retained receipt declares exact reproduction of all 51 approved request-2 prefixes before mutation; review did not rerun the model.',
  intervention: 'Only post-update state is converted F64->nearestF32->F64 in a copied function; occurs before current-token output and next-token recurrence.',
  scopeLimits: [
    'Real shaders also store decayed state before computing memory; this intervention omits that rounding and other F32 arithmetic.',
    'Unchanged evaluator order and shape do not make the diagnostic equivalent to prefill/decode shader arithmetic.',
    'This is a causal storage intervention, not a lower bound, global impossibility proof, approved-reference replacement or precision change.',
    'Original attribution omitted full forced-prefix binding; e148f2ed adds archive/manifest/history/step/fullPrefixes checks and retains a separate validation receipt for the unchanged numerical result.',
  ],
  observed: { prefixesOverTolerance: control.prefixesOverTolerance, maximum: control.comparison.max,
    runtimeArchive: comparison.archiveSha256, signedErrorCorrelation: comparison.signedErrorCorrelation,
    actualTail: comparison.actualTail, controlTail: comparison.controlTail,
    tailOverlap: comparison.tailOverlap, sameSignTailOverlap: comparison.sameSignTailOverlap,
    actualAgainstControlMaximum: comparison.actualAgainstControlled.max },
  conclusion: 'Partial signed-error alignment supports investigating state storage. Unexplained errors remain; no tolerance or authority changes follow.',
};
fs.writeFileSync(new URL('recurrent-storage-independent-review.json', import.meta.url), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ passed: true, roundingCases: 60, scope: 'independent diagnostic review only' }));
