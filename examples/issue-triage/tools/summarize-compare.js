// Summarize tools/product-compare.js output: quality on the held-out test sample,
// plus acquisition, opening, latency, memory, cancellation and recovery per engine.
import fs from 'node:fs/promises';
import path from 'node:path';

const [resultDir, installPath, outputPath] = process.argv.slice(2);
if (!outputPath) throw new Error('usage: summarize-compare.js <resultDir> <install.json> <summary.json>');
const LABELS = ['bug', 'feature', 'question'];
const files = (await fs.readdir(resultDir)).filter((f) => /^(doppler|transformersjs)-\d+\.json$/.test(f));
const runs = await Promise.all(files.map(async (f) => JSON.parse(await fs.readFile(path.join(resultDir, f), 'utf8'))));
const median = (v) => { const s = [...v].sort((a, b) => a - b); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const quantile = (v, q) => [...v].sort((a, b) => a - b)[Math.min(v.length - 1, Math.floor(q * v.length))];
const score = (row, id) => row.routes.find((r) => r.id === id).score;

function predict(rows, offsets) {
  return rows.map((row) => LABELS.reduce((best, id) => (score(row, id) - offsets[id] > score(row, best) - offsets[best] ? id : best), LABELS[0]));
}
function metrics(rows, predicted) {
  const accuracy = rows.filter((row, i) => predicted[i] === row.label).length / rows.length;
  const f1 = LABELS.map((label) => {
    const tp = rows.filter((r, i) => r.label === label && predicted[i] === label).length;
    const fp = rows.filter((r, i) => r.label !== label && predicted[i] === label).length;
    const fn = rows.filter((r, i) => r.label === label && predicted[i] !== label).length;
    return tp ? (2 * tp) / (2 * tp + fp + fn) : 0;
  });
  return { accuracy, macroF1: f1.reduce((a, b) => a + b, 0) / 3, f1: Object.fromEntries(LABELS.map((l, i) => [l, f1[i]])) };
}
function rng(seed) { let x = seed >>> 0; return () => { x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; }; }

const summary = { engines: {}, install: JSON.parse(await fs.readFile(installPath, 'utf8')) };
const predictions = {};
for (const engine of ['doppler', 'transformersjs']) {
  const mine = runs.filter((r) => r.engine === engine).sort((a, b) => a.rep - b.rep);
  if (!mine.length) continue;
  const q = mine.find((r) => r.quality)?.quality;
  const out = { repetitions: mine.length };
  if (q) {
    const offsets = Object.fromEntries(LABELS.map((id) => [id, q.train.reduce((a, r) => a + score(r, id), 0) / q.train.length]));
    const zero = Object.fromEntries(LABELS.map((id) => [id, 0]));
    const raw = predict(q.test, zero);
    const centered = predict(q.test, offsets);
    predictions[engine] = { rows: q.test, raw, centered };
    out.quality = {
      testIssues: q.test.length,
      raw: metrics(q.test, raw),
      labelFreeCentered: { ...metrics(q.test, centered), offsetsFromUnlabeledTrain: offsets },
      testMsPerIssue: { median: median(q.test.map((r) => r.elapsedMs)), p90: quantile(q.test.map((r) => r.elapsedMs), 0.9) },
    };
  }
  const warm = mine.flatMap((r) => r.warmMs);
  out.coldOpenMs = median(mine.map((r) => r.cold.openMs));
  out.coldOpenMsAll = mine.map((r) => r.cold.openMs);
  out.firstDecisionMs = median(mine.map((r) => r.cold.firstDecisionMs));
  out.launchToFirstDecisionMs = median(mine.map((r) => r.cold.launchToFirstDecisionMs));
  out.acquiredModelBytes = median(mine.map((r) => r.cold.transfer.modelBytes));
  out.warmDecisionMs = { median: median(warm), p90: quantile(warm, 0.9), samples: warm.length };
  out.peakMemoryBytes = Object.fromEntries(Object.keys(mine[0].cold.peakMemory).map((k) => [k, median(mine.map((r) => r.cold.peakMemory[k]))]));
  out.cancellation = mine.map((r) => ({ status: r.cancel.status, code: r.cancel.code ?? null, abortToSettleMs: r.cancel.abortToSettleMs, reuseIdentical: r.cancel.reuseIdentical }));
  out.offlineReopen = mine.map((r) => (r.offline.failed
    ? { failed: true, blockedUrls: r.offline.transfer?.blockedUrls, cacheWarning: (r.cold.pageErrors ?? []).find((e) => /browser cache/.test(e)) ?? null }
    : { openMs: r.offline.openMs, identical: r.offline.identical, modelBytes: r.offline.transfer.modelBytes }));
  out.crashRecovery = mine.map((r) => (r.recovery.failed ? { failed: true } : r.recovery));
  out.retainedStorage = mine[0].cold.storage ? { usage: mine[0].cold.storage.usage, cachedEntries: mine[0].cold.storage.cached.length } : null;
  summary.engines[engine] = out;
}
if (predictions.doppler && predictions.transformersjs) {
  const d = predictions.doppler, t = predictions.transformersjs;
  const pair = (kind) => {
    const diffs = d.rows.map((row, i) => (d[kind][i] === row.label ? 1 : 0) - (t[kind][i] === row.label ? 1 : 0));
    const next = rng(20261022);
    const boots = [];
    for (let b = 0; b < 10000; b += 1) { let s = 0; for (let i = 0; i < diffs.length; i += 1) s += diffs[Math.floor(next() * diffs.length)]; boots.push(s / diffs.length); }
    boots.sort((a, b) => a - b);
    return { accuracyDifference: diffs.reduce((a, b) => a + b, 0) / diffs.length, bootstrap95: [boots[250], boots[9750]],
      agreement: d.rows.filter((_, i) => d[kind][i] === t[kind][i]).length };
  };
  summary.dopplerMinusTransformersjs = { raw: pair('raw'), labelFreeCentered: pair('centered') };
}
await fs.writeFile(outputPath, `${JSON.stringify(summary, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
