// Summarize decision-benchmark runs (contract: reports/decision-benchmark-20261022/contract.json).
//   node summarize.js <resultDir> <summary.json>
import fs from 'node:fs/promises';
import path from 'node:path';

const [resultDir, outputPath] = process.argv.slice(2);
const LABELS = ['bug', 'feature', 'question'];
const PRODUCT_DOPPLER = { node: 'doppler-q4k' };
const PRODUCT_BASE = 'doppler-capsule-q4k';

const median = (v) => { if (!v.length) return null; const s = [...v].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const quantile = (v, q) => { if (!v.length) return null; const s = [...v].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
const softmax = (v) => { const m = Math.max(...v); const e = v.map((x) => Math.exp(x - m)); const t = e.reduce((a, b) => a + b, 0); return e.map((x) => x / t); };

function quality(rows, probsOf) {
  const preds = rows.map((r) => { const p = probsOf(r); const choice = LABELS.reduce((b, id) => (p[id] > p[b] ? id : b), LABELS[0]); return { label: r.label, p, choice }; });
  const accuracy = preds.filter((x) => x.choice === x.label).length / preds.length;
  const f1 = Object.fromEntries(LABELS.map((id) => {
    const tp = preds.filter((x) => x.label === id && x.choice === id).length;
    const fp = preds.filter((x) => x.label !== id && x.choice === id).length;
    const fn = preds.filter((x) => x.label === id && x.choice !== id).length;
    return [id, tp ? (2 * tp) / (2 * tp + fp + fn) : 0];
  }));
  // Calibration of the three-way distribution: top-choice ECE over 10 equal-width bins,
  // multiclass Brier score, and negative log-likelihood of the true label.
  const bins = Array.from({ length: 10 }, () => ({ n: 0, conf: 0, correct: 0 }));
  let brier = 0;
  let nll = 0;
  for (const x of preds) {
    const conf = x.p[x.choice];
    const bin = bins[Math.min(9, Math.floor(conf * 10))];
    bin.n += 1; bin.conf += conf; bin.correct += x.choice === x.label ? 1 : 0;
    brier += LABELS.reduce((a, id) => a + (x.p[id] - (id === x.label ? 1 : 0)) ** 2, 0);
    nll -= Math.log(Math.max(1e-12, x.p[x.label]));
  }
  const ece = bins.reduce((a, b) => a + (b.n ? Math.abs(b.correct / b.n - b.conf / b.n) * b.n : 0), 0) / preds.length;
  return { n: preds.length, accuracy, macroF1: (f1.bug + f1.feature + f1.question) / 3, f1, ece, brier: brier / preds.length, nll: nll / preds.length,
    meanConfidence: preds.reduce((a, x) => a + x.p[x.choice], 0) / preds.length, choices: preds.map((x) => x.choice) };
}

function rng(seed) {
  let x = seed >>> 0;
  return () => { x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; };
}
function pairedBootstrap(a, b, labels) {
  const diff = a.map((c, i) => (c === labels[i] ? 1 : 0) - (b[i] === labels[i] ? 1 : 0));
  const next = rng(20261022);
  const means = [];
  for (let k = 0; k < 10000; k += 1) {
    let s = 0;
    for (let i = 0; i < diff.length; i += 1) s += diff[Math.floor(next() * diff.length)];
    means.push(s / diff.length);
  }
  means.sort((x, y) => x - y);
  return { difference: diff.reduce((x, y) => x + y, 0) / diff.length, ci95: [means[249], means[9749]], agree: a.filter((c, i) => c === b[i]).length };
}

const records = [];
for (const platform of await fs.readdir(resultDir, { withFileTypes: true })) {
  if (!platform.isDirectory() || platform.name === 'profiles') continue;
  for (const file of await fs.readdir(path.join(resultDir, platform.name))) {
    if (file.endsWith('.json')) records.push(JSON.parse(await fs.readFile(path.join(resultDir, platform.name, file), 'utf8')));
  }
}

const lanes = {};
for (const r of records) {
  const key = `${r.browser.name}/${r.lane}`;
  lanes[key] ??= { platform: r.browser.name, lane: r.lane, runs: [], cpu: null };
  if (r.mode === 'cpu') lanes[key].cpu = r; else lanes[key].runs.push(r);
}

const summary = { lanes: {}, comparisons: {} };
for (const [key, entry] of Object.entries(lanes)) {
  const runs = entry.runs.sort((a, b) => a.rep - b.rep);
  const ok = runs.filter((r) => !r.report.error);
  const q = runs.find((r) => r.quality && !r.report.error)?.report ?? null;
  const out = {
    platform: entry.platform, lane: entry.lane,
    browserVersion: runs[0]?.browser.version ?? null, userAgent: runs[0]?.report.userAgent ?? null,
    errors: runs.filter((r) => r.report.error).map((r) => `rep${r.rep}: ${r.report.error.split('\n')[0]}`),
    runtime: ok[0]?.report.runtime ?? null, gpu: runs[0]?.report.gpu ?? null,
    coldCompleteMs: ok.map((r) => r.report.coldCompleteMs), openMs: ok.map((r) => r.report.openMs), firstDecisionMs: ok.map((r) => r.report.firstDecisionMs),
    peakRss: ok.map((r) => r.peakMemory?.totalRss ?? null), peakGpuMemory: ok.map((r) => r.peakMemory?.gpuMemory ?? null),
    modelBytes: ok.map((r) => r.transfer?.modelBytes ?? null),
    messages: [...new Set(runs.flatMap((r) => r.report.messages ?? []))].slice(0, 12),
  };
  if (q?.test?.length) {
    const warm = q.test.map((d) => d.elapsedMs);
    out.warmDecisionMs = { median: median(warm), p90: quantile(warm, 0.9), samples: warm.length };
    out.test = quality(q.test, (d) => d.probabilities);
    if (q.train?.length) out.train = quality(q.train, (d) => d.probabilities);
    if (q.test[0]?.detail?.routes && q.train?.length) {
      const score = (d, id) => d.detail.routes.find((r) => r.id === id).score;
      const offsets = Object.fromEntries(LABELS.map((id) => [id, q.train.reduce((a, d) => a + score(d, id), 0) / q.train.length]));
      const centered = (d) => { const p = softmax(LABELS.map((id) => score(d, id) - offsets[id])); return Object.fromEntries(LABELS.map((id, i) => [id, p[i]])); };
      out.testLabelFreeCentered = { ...quality(q.test, centered), offsets };
    }
    // Later repetitions must reproduce the first 10 decisions of the quality run.
    const base = q.test.slice(0, 10);
    out.determinism = ok.filter((r) => r.report !== q).map((r) => ({
      rep: r.rep,
      sameChoices: (r.report.test ?? []).slice(0, 10).every((d, i) => d.choice === base[i]?.choice),
      maxProbabilityDifference: Math.max(0, ...(r.report.test ?? []).slice(0, 10).flatMap((d, i) => LABELS.map((id) => Math.abs(d.probabilities[id] - base[i].probabilities[id])))),
    }));
  }
  if (entry.cpu) {
    const c = entry.cpu.report;
    out.cpuFallback = c.error
      ? { status: 'failed', error: c.error.split('\n')[0], gpuRemoved: c.gpuRemoved ?? null }
      : { status: 'ran', runtime: c.runtime, test: quality(c.test, (d) => d.probabilities), warmMedianMs: median(c.test.map((d) => d.elapsedMs)),
        coldCompleteMs: c.coldCompleteMs, peakRss: entry.cpu.peakMemory?.totalRss ?? null };
    if (out.cpuFallback.test) delete out.cpuFallback.test.choices;
  }
  summary.lanes[key] = out;
}

for (const platform of new Set(Object.values(summary.lanes).map((l) => l.platform))) {
  const get = (lane) => summary.lanes[`${platform}/${lane}`];
  const labelsOf = (lane) => lanes[`${platform}/${lane}`].runs.find((r) => r.quality && !r.report.error)?.report.test.map((d) => d.label);
  const base = get(PRODUCT_DOPPLER[platform] ?? PRODUCT_BASE);
  const product = {};
  const openJev = ['open-jev-kev-0.6b', 'open-jev-kev-4b', 'open-jev-deberta'].filter((l) => get(l)?.test);
  for (const lane of openJev) {
    if (base?.test) product[lane] = { minusDoppler: pairedBootstrap(get(lane).test.choices, base.test.choices, labelsOf(lane)) };
  }
  const best = openJev.filter((l) => get(l).train).sort((a, b) => get(b).train.accuracy - get(a).train.accuracy)[0] ?? null;
  const parity = get('doppler-f16')?.test && get('transformersjs-fp16')?.test
    ? { dopplerF16MinusTransformersjsFp16: pairedBootstrap(get('doppler-f16').test.choices, get('transformersjs-fp16').test.choices, labelsOf('doppler-f16')) }
    : null;
  summary.comparisons[platform] = { productBaseline: base ? base.lane : null, bestOpenAlternativeByTrainAccuracy: best, product, parity };
}
for (const lane of Object.values(summary.lanes)) {
  for (const k of ['test', 'train', 'testLabelFreeCentered']) if (lane[k]) delete lane[k].choices;
}
await fs.writeFile(outputPath, `${JSON.stringify(summary, null, 1)}\n`);

const fmt = (v, d = 0) => (v == null ? '–' : Number(v).toFixed(d));
const gib = (v) => (v == null ? '–' : (v / 2 ** 30).toFixed(2));
console.log('| platform | lane | runtime | test acc | macro-F1 | ECE | Brier | cold ms (median) | warm median/p90 ms | peak RSS GiB | GPU mem GiB | MB fetched |');
console.log('|---|---|---|---|---|---|---|---|---|---|---|---|');
for (const l of Object.values(summary.lanes).sort((a, b) => (a.platform + a.lane).localeCompare(b.platform + b.lane))) {
  const rt = l.runtime ? [l.runtime.device ?? l.runtime.surface, l.runtime.dtype ?? l.runtime.weights].filter(Boolean).join(' ') : `failed: ${l.errors[0]?.slice(0, 60)}`;
  console.log(`| ${l.platform} | ${l.lane} | ${rt} | ${fmt(l.test?.accuracy, 3)} | ${fmt(l.test?.macroF1, 3)} | ${fmt(l.test?.ece, 3)} | ${fmt(l.test?.brier, 3)} | ${fmt(median(l.coldCompleteMs))} | `
    + `${fmt(l.warmDecisionMs?.median)}/${fmt(l.warmDecisionMs?.p90)} | ${gib(median(l.peakRss))} | ${gib(median(l.peakGpuMemory.filter((v) => v != null)))} | ${fmt(median(l.modelBytes.filter((v) => v != null)) / 1e6)} |`);
}
