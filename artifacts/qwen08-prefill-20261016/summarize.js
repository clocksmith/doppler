// Run from the repository root after paired-run.js and both boundary captures.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

const directory = new URL('./', import.meta.url);
const read = async (file) => JSON.parse(await fs.readFile(new URL(file, directory)));
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2;
};
const receipt = await read('paired-receipt.json');
assert.equal(receipt.complete, true);
const reference = await read('../qwen08-m3-mlx-20261015/inputs.json');
const timing = [];
let exactTokenRuns = 0;
for (const promptTokens of [128, 1024]) {
  const samples = { baseline: [], candidate: [] };
  for (const name of Object.keys(samples)) {
    for (let pair = 0; pair < 4; pair++) {
      const result = await read(`paired-${pair}-${name}.json`);
      assert.equal(result.complete, true);
      const runs = result.runs.filter((run) => run.promptTokenIds.length === promptTokens);
      assert.equal(runs.length, 2);
      for (const run of runs) {
        assert.equal(run.evidence.tokenIds.length, 128);
        assert.deepEqual(run.promptTokenIds, reference.prompts.find((p) => p.tokenIds.length === promptTokens).tokenIds);
        const baseline = await read('paired-0-baseline.json');
        assert.deepEqual(run.evidence.tokenIds, baseline.runs.find((r) => r.promptTokenIds.length === promptTokens).evidence.tokenIds);
        exactTokenRuns++;
        if (!run.warmup) samples[name].push({ pair, totalMs: run.totalRunMs, firstTokenMs: run.firstTokenMs });
      }
    }
  }
  const medians = Object.fromEntries(Object.entries(samples).map(([name, rows]) => [name, {
    totalMs: median(rows.map((r) => r.totalMs)), firstTokenMs: median(rows.map((r) => r.firstTokenMs)),
  }]));
  timing.push({ promptTokens, outputTokens: 128, samples, medians,
    totalReductionPercent: 100 * (1 - medians.candidate.totalMs / medians.baseline.totalMs),
    firstTokenReductionPercent: 100 * (1 - medians.candidate.firstTokenMs / medians.baseline.firstTokenMs) });
}

const baseline = await read('boundary-baseline.json');
const candidate = await read('boundary-candidate.json');
assert.equal(baseline.complete, true);
assert.equal(candidate.complete, true);
const numerical = [];
const absoluteTolerance = 1e-3;
for (let index = 0; index < baseline.runs.length; index++) {
  const a = baseline.runs[index], b = candidate.runs[index];
  assert.deepEqual(a.promptTokenIds, b.promptTokenIds);
  assert.deepEqual(a.evidence.tokenIds, b.evidence.tokenIds);
  const logits = [];
  assert.equal(a.logitsFiles.length, 2);
  assert.equal(b.logitsFiles.length, 2);
  for (let step = 0; step < a.logitsFiles.length; step++) {
    const records = [a.logitsFiles[step], b.logitsFiles[step]];
    const bytes = await Promise.all(records.map(async (record) => {
      const value = await fs.readFile(record.file);
      assert.equal(createHash('sha256').update(value).digest('hex'), record.sha256);
      assert.equal(value.length, record.count * 4);
      return value;
    }));
    assert.equal(bytes[0].length, bytes[1].length);
    let maxAbs = 0;
    for (let offset = 0; offset < bytes[0].length; offset += 4) {
      const x = bytes[0].readFloatLE(offset), y = bytes[1].readFloatLE(offset);
      assert.ok(!Number.isNaN(x) && !Number.isNaN(y));
      if (!Number.isFinite(x) || !Number.isFinite(y)) assert.equal(x, y);
      else maxAbs = Math.max(maxAbs, Math.abs(x - y));
    }
    assert.ok(maxAbs <= absoluteTolerance);
    logits.push({ step, values: records[0].count, maxAbs, files: records });
  }
  const at = a.evidence.stats.operatorDiagnostics.timeline;
  const bt = b.evidence.stats.operatorDiagnostics.timeline;
  assert.equal(at.length, bt.length);
  let maxSampleAbs = 0, sampleValues = 0;
  const layers = new Set();
  for (let op = 0; op < at.length; op++) {
    const x = at[op], y = bt[op];
    assert.equal(x.opId, y.opId);
    assert.equal(x.shapeSignature, y.shapeSignature);
    assert.equal(x.dtype, y.dtype);
    if (x.layerIndex != null) layers.add(x.layerIndex);
    assert.deepEqual(x.capture.sampleCoordinates, y.capture.sampleCoordinates);
    assert.equal(x.capture.hasNaN, false);
    assert.equal(y.capture.hasNaN, false);
    assert.equal(x.capture.hasInf, y.capture.hasInf);
    assert.equal(x.capture.sample.length, y.capture.sample.length);
    for (let sample = 0; sample < x.capture.sample.length; sample++) {
      const u = x.capture.sample[sample], v = y.capture.sample[sample];
      if (u == null || v == null) assert.equal(u, v);
      else { maxSampleAbs = Math.max(maxSampleAbs, Math.abs(u - v)); sampleValues++; }
    }
  }
  assert.equal(layers.size, 24);
  assert.ok(maxSampleAbs <= absoluteTolerance);
  numerical.push({ promptTokens: a.promptTokenIds.length, logits, captures: at.length,
    sampledLayers: layers.size, sampleValues, maxSampleAbs });
}

const summary = {
  scope: 'Local Node/Dawn M3 diagnostic: Qwen3.5-0.8B F16 weights, F32 activations, F16 KV. Not vendor qualification.',
  profile: 'profiles/qwen-3-5-0-8b-f16-m3', runtimeBase: receipt.plan.runtimeBase,
  runtimePatch: 'runtime.patch', pairedReceipt: 'paired-receipt.json',
  methodology: receipt.plan.timing, exactTokenRuns, timing,
  numerical: { absoluteTolerance, scope: 'Full logits at two steps per prompt; eight sampled values per captured operator, not full activation parity.', rows: numerical },
  selected: ['Register-tiled F16-weight/F32-activation prefill matmul', 'Head-256 decode attention', 'LM-head subgroup geometry: 8 columns, 32 threads per column'],
  rejected: [
    { candidate: 'FlashAttention prefill', reason: 'Repaired nine-binding dispatch and resource ownership; correct CPU reference, slower model answers. Not selected.' },
    { candidate: 'Coalesced shared-memory load mapping', reason: 'Bit-identical standalone output, no consistent timing improvement. Shader reverted.' },
    { candidate: 'Larger K staging tiles', reason: 'Valid K32/K64 probes slower. Initial invalid-pipeline result retained in tile-depth-invalid.json and excluded; final probe validates GPU errors and poisons output. Intermediate poison-write usage error retained in tile-depth-with-invalid-poison.json; corrected final rerun validates the write too. Shader reverted.' },
    { candidate: 'Fused recurrent decode core', reason: 'No clear complete-answer improvement; not selected.' },
  ],
  limitations: ['Four timed samples per configuration and prompt length; all retained including the last slow baseline.',
    'Short-prompt complete-answer median essentially unchanged.',
    'No new MLX or Doe/Dawn comparison; no claim of beating either competitor.',
    'Opt-in profile for the pinned local F16 text artifact; no default, browser, quantized, or other-model promotion.'],
};
await fs.writeFile(new URL('summary.json', directory), JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({ exactTokenRuns, timing: timing.map(({ promptTokens, medians, totalReductionPercent }) => ({ promptTokens, medians, totalReductionPercent })), numerical: numerical.map(({ promptTokens, maxSampleAbs, logits }) => ({ promptTokens, maxSampleAbs, logitsMaxAbs: logits.map((r) => r.maxAbs) })) }, null, 2));
