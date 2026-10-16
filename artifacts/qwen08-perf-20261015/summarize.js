// Recompute diagnostic comparisons without running a GPU workload.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { compare } from '../qwen08-m3-mlx-20261015/compare.js';

const read = async name => JSON.parse(await fs.readFile(new URL(name, import.meta.url)));
const median = values => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const baseline = await read('baseline.json');
const comparisons = {};
for (const name of ['baseline-confirm', 'tiled', 'tiled-confirm', 'doe-complete']) {
  comparisons[name] = compare(baseline, await read(`${name}.json`));
}
const cancellation = await read('doe-cancel-reuse.json');
assert.equal(cancellation.complete, true);
assert.equal(cancellation.cancelled.tokenCount, 3);
assert.deepEqual(cancellation.reuseTokenIds, baseline.runs[0].evidence.tokenIds.slice(0, 16));
const stats = (await read('gpu-timestamps.json')).runs[0].evidence.stats;
const wasm = await read('wasm-probe.json');
assert.equal(wasm.checkedHalfPatterns, 65536);
assert.equal(wasm.tailChecked, true);
const mma = await read('mma-probe-all.json');
assert.equal(mma.rows.length, 12);
const summary = {
  scope: 'M3 Qwen3.5-0.8B F16 diagnostics; no vendor ranking or runtime optimization promotion',
  runtimeCommit: (await fs.readFile(new URL('runtime-commit.txt', import.meta.url), 'utf8')).trim(),
  comparisons,
  dispositions: {
    tiledPrefill: 'Not promoted. Initial improvement did not reproduce; confirmation regressed. Host contention is recorded, not corrected by excluding slow samples.',
    doe: 'Correctness fixes retained. All complete outputs match Dawn; cancellation and reuse pass. These sequential timings do not establish a performance advantage.',
    wasm: 'Exact F16 materialization prototype only. Includes optimized JavaScript control and copies. Not integrated into inference; profile does not identify this as the dominant cost.',
    webnn: 'Flagged browser graph probe only. Actual backend identity, ANE use, power and concurrent GPU availability remain unverified.',
    subgroupMatrix: 'Correct synthetic projections, no consistent improvement over existing tiled kernel. Not integrated into model execution.',
    batching: 'Existing four-token mode tested in adjacent baseline artifacts. Not a new zero-readback implementation and not promoted.',
  },
  profile: {
    prefillWallMs: stats.prefillTimeMs,
    prefillGpuMs: stats.gpuTimePrefillMs,
    decodeWallMs: stats.decodeTimeMs,
    decodeGpuMs: stats.gpuTimeDecodeMs,
    caveat: 'Instrumented 1024-prompt/16-output run; overlapping waits and CPU samples are not an additive latency budget.',
  },
  wasm: wasm.rows.map(row => ({
    count: row.count,
    medianMs: Object.fromEntries(Object.entries(row.times).map(([key, values]) => [key, median(values)])),
    optimizedJsToWasmWithCopiesRatio: median(row.times.jsBitConversion) / median(row.times.wasmWithCopies),
  })),
  subgroupMatrix: mma.rows.map(({M, N, K, name, times, maxAbsVsReference}) => ({M, N, K, name, medianMs: median(times), maxAbsVsReference})),
  component: 'doppler',
  intent: 'preserved',
  boundaryEffects: 'Evidence and standalone probes only. Production fixes are owned by Doe native Metal and WGSL lowering.',
};
await fs.writeFile(new URL('summary.json', import.meta.url), `${JSON.stringify(summary, null, 2)}\n`);
console.log('Complete-output agreement, cancellation/reuse, and diagnostic summary verified.');
