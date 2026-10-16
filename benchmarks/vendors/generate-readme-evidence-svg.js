#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const RATE_FORMATTER = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
  useGrouping: false,
});

const GROUPS = Object.freeze([
  Object.freeze({
    label: 'Apple M3 · Metal',
    rows: Object.freeze([
      Object.freeze({
        model: 'Qwen 3.5 0.8B text',
        kind: 'text',
        receipt: 'benchmarks/vendors/results/compare_20260709T154633.json',
        section: 'parity',
      }),
      Object.freeze({
        model: 'Qwen 3 Embedding 0.6B',
        kind: 'embedding',
        receipt: 'benchmarks/vendors/results/embedding_compare_qwen-3-embedding-0-6b-q4k-ehf16-af32_20260709T180853.json',
      }),
      Object.freeze({
        model: 'Qwen 3 Reranker 0.6B',
        kind: 'rerank',
        receipt: 'benchmarks/vendors/results/rerank_compare_qwen-3-reranker-0-6b-q4k-ehf16-af32_20260709T192830.json',
      }),
      Object.freeze({
        model: 'Qwen 3.5 2B text',
        kind: 'paired-text',
        receipt: 'benchmarks/vendors/results/qwen35-2b-metal-paired-p256-d512-20260710.json',
        detail: 'local model',
      }),
    ]),
  }),
  Object.freeze({
    label: 'Radeon 8060S · Vulkan',
    rows: Object.freeze([
      Object.freeze({
        model: 'Qwen 3.5 0.8B text',
        kind: 'text',
        receipt: 'benchmarks/vendors/results/compare_20260707T153509.json',
        section: 'throughput',
      }),
      Object.freeze({
        model: 'Qwen 3 Embedding 0.6B',
        kind: 'embedding',
        receipt: 'benchmarks/vendors/results/embedding_compare_qwen-3-embedding-0-6b-q4k-ehf16-af32_20260710T011455.json',
      }),
      Object.freeze({
        model: 'Qwen 3 Reranker 0.6B',
        kind: 'rerank',
        receipt: 'benchmarks/vendors/results/rerank_compare_qwen-3-reranker-0-6b-q4k-ehf16-af32_20260710T014450.json',
      }),
      Object.freeze({
        model: 'Qwen 3.5 2B text',
        kind: 'text',
        receipt: 'benchmarks/vendors/results/compare_20260707T161623.json',
        section: 'throughput',
        detail: 'local model',
      }),
      Object.freeze({
        model: 'Gemma 4 E2B INT4-PLE',
        kind: 'text',
        receipt: 'benchmarks/vendors/results/compare_20260707T170557.json',
        section: 'parity',
        detail: 'local model',
      }),
    ]),
  }),
]);

function readJson(relativePath) {
  return JSON.parse(fs.readFileSync(path.join(REPO_ROOT, relativePath), 'utf8'));
}

function requiredNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`missing positive finite ${label}`);
  }
  return value;
}

function requiredArray(value, label) {
  if (!Array.isArray(value) || value.length === 0) throw new Error(`missing non-empty ${label}`);
  return value;
}

function rangeFromValues(values, label) {
  const finite = requiredArray(values, label).map((value, index) => requiredNumber(value, `${label}[${index}]`));
  return {
    min: Math.min(...finite),
    max: Math.max(...finite),
  };
}

function rangeFromLatency(minMs, maxMs, label) {
  const minimumLatency = requiredNumber(minMs, `${label}.minMs`);
  const maximumLatency = requiredNumber(maxMs, `${label}.maxMs`);
  if (minimumLatency <= 0 || maximumLatency <= 0) throw new Error(`${label} latency must be positive`);
  return {
    min: 1000 / maximumLatency,
    max: 1000 / minimumLatency,
  };
}

function loadTextRow(definition, report) {
  const section = report?.sections?.compute?.[definition.section];
  if (!section) throw new Error(`${definition.receipt} has no ${definition.section} section`);
  const dopplerMetrics = section?.doppler?.result?.metrics;
  const dopplerStats = dopplerMetrics?.throughput?.decodeTokensPerSec;
  const tjsMetrics = section?.transformersjs?.metrics;
  const tjsRuns = requiredArray(section?.transformersjs?.runs, `${definition.receipt}.transformersjs.runs`);
  return {
    ...definition,
    unit: 'tok/s',
    runs: requiredNumber(report.runs, `${definition.receipt}.runs`),
    doppler: {
      rate: requiredNumber(dopplerMetrics?.decodeTokensPerSec, `${definition.receipt}.doppler.decodeTokensPerSec`),
      min: requiredNumber(dopplerStats?.min, `${definition.receipt}.doppler.min`),
      max: requiredNumber(dopplerStats?.max, `${definition.receipt}.doppler.max`),
    },
    transformersjs: {
      rate: requiredNumber(tjsMetrics?.decodeTokensPerSec, `${definition.receipt}.transformersjs.decodeTokensPerSec`),
      ...rangeFromValues(
        tjsRuns.map((run) => run?.decodeTokensPerSec),
        `${definition.receipt}.transformersjs.decodeTokensPerSec`,
      ),
    },
  };
}

function loadPairedTextRow(definition, report) {
  return {
    ...definition,
    unit: 'tok/s',
    runs: requiredArray(report.pairs, `${definition.receipt}.pairs`).length,
    doppler: {
      rate: requiredNumber(report?.results?.doppler?.medianTokensPerSec, `${definition.receipt}.doppler.medianTokensPerSec`),
      min: requiredNumber(report?.results?.doppler?.minTokensPerSec, `${definition.receipt}.doppler.minTokensPerSec`),
      max: requiredNumber(report?.results?.doppler?.maxTokensPerSec, `${definition.receipt}.doppler.maxTokensPerSec`),
    },
    transformersjs: {
      rate: requiredNumber(report?.results?.transformersjs?.medianTokensPerSec, `${definition.receipt}.transformersjs.medianTokensPerSec`),
      min: requiredNumber(report?.results?.transformersjs?.minTokensPerSec, `${definition.receipt}.transformersjs.minTokensPerSec`),
      max: requiredNumber(report?.results?.transformersjs?.maxTokensPerSec, `${definition.receipt}.transformersjs.maxTokensPerSec`),
    },
    pairWins: {
      doppler: requiredNumber(report?.results?.comparison?.dopplerPairWins, `${definition.receipt}.dopplerPairWins`),
      transformersjs: requiredNumber(report?.results?.comparison?.transformersjsPairWins, `${definition.receipt}.transformersjsPairWins`),
    },
  };
}

function loadRetrievalRow(definition, report) {
  const isEmbedding = definition.kind === 'embedding';
  const rateField = isEmbedding ? 'avgEmbeddingsPerSec' : 'avgReranksPerSec';
  const minLatencyField = isEmbedding ? 'minEmbeddingMs' : 'minRerankMs';
  const maxLatencyField = isEmbedding ? 'maxEmbeddingMs' : 'maxRerankMs';
  const timedRuns = report?.raw?.dopplerBench?.metrics?.timedRuns;
  const dopplerMetrics = report?.raw?.dopplerBench?.metrics;
  const tjsMetrics = report?.raw?.transformersjsBench?.metrics;
  return {
    ...definition,
    unit: isEmbedding ? 'embedding/s' : 'reranks/s',
    runs: requiredNumber(timedRuns, `${definition.receipt}.timedRuns`),
    doppler: {
      rate: requiredNumber(report?.summary?.doppler?.speed?.[rateField], `${definition.receipt}.doppler.${rateField}`),
      ...rangeFromLatency(
        dopplerMetrics?.[minLatencyField],
        dopplerMetrics?.[maxLatencyField],
        `${definition.receipt}.doppler`,
      ),
    },
    transformersjs: {
      rate: requiredNumber(report?.summary?.transformersjs?.speed?.[rateField], `${definition.receipt}.transformersjs.${rateField}`),
      ...rangeFromLatency(
        tjsMetrics?.[minLatencyField],
        tjsMetrics?.[maxLatencyField],
        `${definition.receipt}.transformersjs`,
      ),
    },
  };
}

export function requireComparable(report, definition) {
  if (definition.kind === 'paired-text') {
    if (report.claimGrade !== true || report.correctness?.exactTokenParityAllPairs !== true
      || report.correctness?.pairedComparableAllPairs !== true) {
      throw new Error(`${definition.receipt}: paired correctness/claim gate failed`);
    }
    return;
  }
  if (report.fairness?.claimGrade !== true || report.fairness?.correctnessOk !== true) {
    throw new Error(`${definition.receipt}: correctness/claim gate failed`);
  }
  if (definition.kind === 'text') {
    const section = report.sections?.compute?.[definition.section];
    if (report.fairness?.surfaceFairness?.sameRuntimeSurface !== true
      || report.fairness?.sections?.[`compute/${definition.section}`]?.claimGrade !== true
      || section?.pairedComparable !== true || section?.decodeValidity?.ok !== true) {
      throw new Error(`${definition.receipt}: comparable surface/section gate failed`);
    }
  }
}

function loadRow(definition) {
  const report = readJson(definition.receipt);
  requireComparable(report, definition);
  let row;
  if (definition.kind === 'text') row = loadTextRow(definition, report);
  else if (definition.kind === 'paired-text') row = loadPairedTextRow(definition, report);
  else row = loadRetrievalRow(definition, report);
  return { ...row, mismatch: report.correctness?.exactMatch === false };
}

function escapeXml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

const PURPLE = '#7542c8';
const MUTED = '#657083';
const RECEIPT_URL = 'https://github.com/clocksmith/doppler/blob/main/';

function text(x, y, value, size = 18, color = '#202735', weight = 400, extra = '') {
  return `<text x="${x}" y="${y}" font-size="${size}" fill="${color}" font-weight="${weight}" ${extra}>${escapeXml(value)}</text>`;
}

function frame(width, height, title, description, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
<title id="title">${escapeXml(title)}</title><desc id="desc">${escapeXml(description)}</desc>
<style>text{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}a:hover text{text-decoration:underline}</style>
<rect x="1" y="1" width="${width - 2}" height="${height - 2}" rx="18" fill="white" stroke="#dce1e9" stroke-width="1.4"/>
${text(36, 43, 'doppler', 24, PURPLE, 700)}
${body.join('\n')}
</svg>\n`;
}

function bar(x, y, width, value, max, color) {
  return `<rect x="${x}" y="${y}" width="${width}" height="14" rx="7" fill="#eef0f4"/>
<rect x="${x}" y="${y}" width="${(width * value / max).toFixed(2)}" height="14" rx="7" fill="${color}"/>`;
}

function receiptGroup(row, content) {
  return `<g class="benchmark-row" data-receipt="${escapeXml(row.receipt)}"><a href="${RECEIPT_URL}${escapeXml(row.receipt)}">${content.join('\n')}</a></g>`;
}

function renderRateCard(row, x, y, width) {
  const max = Math.max(row.doppler.max, row.transformersjs.max, row.doppler.rate, row.transformersjs.rate);
  const ratio = row.doppler.rate / row.transformersjs.rate;
  const detail = row.mismatch
    ? `${row.runs} runs · ${row.detail} · outputs differ`
    : row.kind === 'paired-text'
      ? `${row.runs} paired runs · exact tokens`
      : `${row.runs} runs · ${row.detail || 'published artifact'}`;
  const content = [
    `<rect x="${x}" y="${y}" width="${width}" height="234" rx="12" fill="#f7f3fd" stroke="#e5ddf3"/>`,
    text(x + 22, y + 33, row.model, 21, '#202735', 600),
    text(x + 22, y + 59, detail, 15, MUTED),
    text(x + width - 22, y + 101, `${ratio.toFixed(2)}×`, 28, PURPLE, 650, 'text-anchor="end"'),
    text(x + 22, y + 100, 'Doppler', 17, PURPLE, 600),
    text(x + 22, y + 127, `${RATE_FORMATTER.format(row.doppler.rate)} ${row.unit}`, 18, PURPLE, 600),
    bar(x + 22, y + 139, width - 44, row.doppler.rate, max, PURPLE),
    text(x + 22, y + 184, 'Transformers.js', 17, MUTED, 500),
    text(x + width - 22, y + 184, `${RATE_FORMATTER.format(row.transformersjs.rate)} ${row.unit}`, 18, MUTED, 500, 'text-anchor="end"'),
    bar(x + 22, y + 198, width - 44, row.transformersjs.rate, max, '#8791a2'),
  ];
  return receiptGroup(row, content);
}

function renderRates(groups, mobile) {
  const width = mobile ? 620 : 1200;
  const columns = mobile ? 1 : 2;
  const cardWidth = (width - 72 - (columns - 1) * 20) / columns;
  const body = [text(36, 85, 'Historical throughput measurements.', mobile ? 26 : 30, '#202735', 600),
    text(36, 116, 'Qwen text uses superseded ONNX exports · July 2026', 17, MUTED)];
  let y = 150;
  for (const group of groups) {
    body.push(text(36, y + 23, group.label, 23, '#202735', 600));
    y += 43;
    group.rows.forEach((row, i) => body.push(renderRateCard(row, 36 + (i % columns) * (cardWidth + 20), y + Math.floor(i / columns) * 254, cardWidth)));
    y += Math.ceil(group.rows.length / columns) * 254 + 20;
  }
  const lines = mobile ? [
    'July 2026 · Transformers.js 4.0.1 · zero-based bars',
    'RDRR Q4K vs ONNX q4f16; different optimized formats.',
    'Rates use each receipt’s estimator; synthetic text workloads.',
    'Retrieval passed semantic fixtures, not numerical equivalence.',
    'Local models and differing Gemma outputs are labeled.',
    'Optimized Qwen export comparison remains unmeasured.',
  ] : [
    'July 2026 · Transformers.js 4.0.1 · zero-based bars · RDRR Q4K vs ONNX q4f16 (different optimized formats).',
    'Rates use each receipt’s estimator; synthetic text workloads. Retrieval passed semantic fixtures, not numerical equivalence.',
    'Local models and differing Gemma outputs are labeled. Optimized Qwen export comparison remains unmeasured.',
  ];
  lines.forEach((line, i) => body.push(text(36, y + 12 + i * 25, line, 15, MUTED)));
  return frame(width, y + lines.length * 25 + 25, 'Doppler vs Transformers.js: historical measured throughput', 'Nine historical browser comparisons with per-card zero-based scales. Qwen generation used superseded ONNX exports. Purple is Doppler, gray is Transformers.js. Every card links to its receipt.', body);
}

const LATENCY_ROWS = [
  { model: 'Qwen 3.5 0.8B', platform: 'Radeon 8060S · Vulkan', kind: 'text', section: 'parity', receipt: 'benchmarks/vendors/results/compare_20260707T153509.json' },
  { model: 'Qwen 3.5 2B', platform: 'Radeon 8060S · Vulkan', kind: 'text', section: 'parity', receipt: 'benchmarks/vendors/results/compare_20260707T161623.json' },
  { model: 'Qwen 3.5 0.8B', platform: 'Apple M3 · Metal', kind: 'text', section: 'parity', receipt: 'benchmarks/vendors/results/compare_20260709T154633.json' },
];

const RETRIEVAL_HIGHLIGHTS = [
  { model: 'Qwen3 Embedding 0.6B', label: 'Embeddings', kind: 'embedding', receipt: 'benchmarks/vendors/results/embedding_compare_qwen-3-embedding-0-6b-q4k-ehf16-af32_20260710T011455.json' },
  { model: 'Qwen3 Reranker 0.6B', label: 'Reranking', kind: 'rerank', receipt: 'benchmarks/vendors/results/rerank_compare_qwen-3-reranker-0-6b-q4k-ehf16-af32_20260710T014450.json' },
];

export function loadHighlight(definition, report, catalog) {
  requireComparable(report, definition);
  const target = catalog.models.find((model) => model.modelId === report.model?.dopplerModelId)
    ?.vendorBenchmark?.transformersjs;
  if (report.fairness?.releaseClaimable !== true
    || !target || target.repoId !== report.model?.transformersjsModelId
    || target.dtype !== report.model?.transformersjsDtype) {
    throw new Error(`${definition.receipt}: current comparator/release gate failed`);
  }
  const d = report.raw?.dopplerBench;
  const t = report.raw?.transformersjsBench;
  const dg = d?.deviceInfo?.adapterInfo;
  const tg = t?.deviceInfo;
  if (!dg?.device || dg.device !== tg?.device || dg.vendor !== tg.vendor
    || d?.env?.runtime !== 'browser' || t?.env?.device !== 'webgpu'
    || report.workload?.cacheMode !== 'warm'
    || d?.metrics?.timedRuns !== t?.timedRuns) {
    throw new Error(`${definition.receipt}: paired hardware/workload gate failed`);
  }
  return loadRetrievalRow(definition, report);
}

function renderHighlights(mobile) {
  const width = mobile ? 620 : 1200;
  const columns = mobile ? 1 : 2;
  const cardWidth = (width - 72 - (columns - 1) * 20) / columns;
  const catalog = readJson('models/catalog.json');
  const body = [text(36, 85, 'More search work per second.', mobile ? 27 : 30, '#202735', 600),
    text(36, 116, 'Doppler vs Transformers.js', 19, MUTED)];
  RETRIEVAL_HIGHLIGHTS.forEach((definition, i) => {
    const row = loadHighlight(definition, readJson(definition.receipt), catalog);
    const ratio = row.doppler.rate / row.transformersjs.rate;
    const max = Math.max(row.doppler.rate, row.transformersjs.rate);
    const x = 36 + (i % columns) * (cardWidth + 20);
    const y = 150 + Math.floor(i / columns) * 316;
    body.push(receiptGroup(row, [
      `<rect x="${x}" y="${y}" width="${cardWidth}" height="296" rx="12" fill="#f7f3fd" stroke="#e5ddf3"/>`,
      text(x + 22, y + 34, definition.label, 23, '#202735', 600),
      text(x + cardWidth - 22, y + 34, definition.model, 16, MUTED, 400, 'text-anchor="end"'),
      text(x + 22, y + 114, `${ratio.toFixed(1)}×`, 68, PURPLE, 650),
      text(x + 22, y + 146, 'higher throughput', 18, PURPLE, 500),
      text(x + 22, y + 191, 'Doppler', 17, PURPLE, 600),
      text(x + cardWidth - 22, y + 191, `${row.doppler.rate.toFixed(2)} ${row.unit}`, 18, PURPLE, 600, 'text-anchor="end"'),
      bar(x + 22, y + 204, cardWidth - 44, row.doppler.rate, max, PURPLE),
      text(x + 22, y + 251, 'Transformers.js', 17, MUTED, 500),
      text(x + cardWidth - 22, y + 251, `${row.transformersjs.rate.toFixed(2)} ${row.unit}`, 18, MUTED, 500, 'text-anchor="end"'),
      bar(x + 22, y + 264, cardWidth - 44, row.transformersjs.rate, max, '#8791a2'),
    ]));
  });
  const y = 150 + Math.ceil(RETRIEVAL_HIGHLIGHTS.length / columns) * 316 + 12;
  const lines = mobile ? [
    'Radeon 8060S · Chromium WebGPU · 15 warm runs',
    'July 2026 · Transformers.js 4.x · RDRR vs ONNX',
    'Loading excluded · semantic fixtures passed · linked receipts',
  ] : [
    'Radeon 8060S · Chromium WebGPU · 15 warm runs · July 2026 · Transformers.js 4.x',
    'RDRR vs ONNX · loading excluded · semantic fixtures passed · each card links its receipt',
  ];
  lines.forEach((line, i) => body.push(text(36, y + i * 25, line, 15, MUTED)));
  return frame(width, y + lines.length * 25 + 22, 'Doppler retrieval benchmarks: 5.1x embedding and 1.7x reranking throughput',
    'Recorded July 2026 Radeon 8060S WebGPU results, 15 warm runs. Qwen3 0.6B embeddings: 33.88 versus 6.69 per second. Reranking: 3.59 versus 2.12 per second. Loading excluded; different optimized formats, semantic fixtures passed. Zero-based bars and source links.', body);
}

function renderLatency(mobile) {
  const width = mobile ? 620 : 1200;
  const columns = mobile ? 1 : 3;
  const cardWidth = (width - 72 - (columns - 1) * 20) / columns;
  const body = [text(36, 85, 'Historical Qwen baseline · July 2026', mobile ? 26 : 30, '#202735', 600),
    text(36, 116, '512 prompt tokens → 128 generated · lower time is better', mobile ? 17 : 19, MUTED)];
  LATENCY_ROWS.forEach((row, i) => {
    const report = readJson(row.receipt);
    requireComparable(report, row);
    if (report.correctness?.exactMatch !== true) throw new Error(`${row.receipt}: exact output gate failed`);
    const section = report.sections.compute.parity;
    const d = section.doppler.result.timing;
    const t = section.transformersjs.timing;
    const dv = requiredNumber(d.totalRunMs, 'doppler totalRunMs') / 1000;
    const tv = requiredNumber(t.totalRunMs, 'transformersjs totalRunMs') / 1000;
    const dLoad = requiredNumber(d.modelLoadMs, 'doppler modelLoadMs') / 1000;
    const tLoad = requiredNumber(t.modelLoadMs, 'transformersjs modelLoadMs') / 1000;
    const x = 36 + (i % columns) * (cardWidth + 20);
    const y = 150 + Math.floor(i / columns) * 414;
    const label = report.fairness.releaseClaimable ? 'published artifact' : 'local artifact';
    body.push(receiptGroup(row, [
      `<rect x="${x}" y="${y}" width="${cardWidth}" height="394" rx="12" fill="#f7f3fd" stroke="#e5ddf3"/>`,
      text(x + 22, y + 35, row.model, 23, '#202735', 600),
      text(x + 22, y + 63, row.platform, 16, MUTED),
      text(x + 22, y + 125, `${(tv / dv).toFixed(1)}×`, 54, PURPLE, 650),
      text(x + 22, y + 155, 'observed warm-time ratio', 18, PURPLE, 500),
      text(x + 22, y + 198, 'Doppler', 17, PURPLE, 600),
      text(x + cardWidth - 22, y + 198, `${dv.toFixed(2)} s`, 21, PURPLE, 600, 'text-anchor="end"'),
      bar(x + 22, y + 213, cardWidth - 44, dv, tv, PURPLE),
      text(x + 22, y + 260, 'Transformers.js', 17, MUTED, 500),
      text(x + cardWidth - 22, y + 260, `${tv.toFixed(2)} s`, 21, MUTED, 500, 'text-anchor="end"'),
      bar(x + 22, y + 275, cardWidth - 44, tv, tv, '#8791a2'),
      text(x + 22, y + 323, `Load + run: ${(dLoad + dv).toFixed(2)} vs ${(tLoad + tv).toFixed(2)} s`, 16, '#202735', 500),
      text(x + 22, y + 350, 'Sum of reported load and run times', 14, MUTED),
      text(x + 22, y + 376, `${report.runs} runs · exact output · ${label}`, 14, MUTED),
    ]));
  });
  const y = 150 + Math.ceil(LATENCY_ROWS.length / columns) * 414 + 12;
  const lines = mobile ? [
    'Warm generation includes prompt + decode; excludes load.',
    'July 2026 · Chromium WebGPU · Transformers.js 4.0.1',
    'RDRR Q4K vs ONNX q4f16: different optimized formats.',
    'Synthetic text workloads; receipt timing summaries.',
    'Warm load shown separately; not a cold-start claim.',
    'Superseded by ONNX-OPT; new comparison pending.',
  ] : [
    'Warm generation includes prompt + decode; excludes model load. Warm load is shown separately, not a cold-start claim.',
    'July 2026 · Chromium WebGPU · Transformers.js 4.0.1 · RDRR Q4K vs ONNX q4f16: different optimized formats.',
    'Superseded by ONNX-OPT; new comparison pending. Synthetic workloads; receipt summaries. Click a card for the source.',
  ];
  lines.forEach((line, i) => body.push(text(36, y + i * 25, line, 15, MUTED)));
  return frame(width, y + lines.length * 25 + 22, 'Historical Qwen times against superseded ONNX exports', 'Three historical same-device browser comparisons with exact output matches. Their superseded ONNX baseline does not establish an advantage over optimized exports. Zero-based time bars; warm loading shown separately.', body);
}

export function main() {
  const unknownArgs = process.argv.slice(2).filter((arg) => arg !== '--check');
  if (unknownArgs.length > 0) throw new Error(`unknown argument: ${unknownArgs[0]}`);
  const groups = GROUPS.map((group) => ({ ...group, rows: group.rows.map(loadRow) }));
  const outputs = [
    ['assets/doppler-webgpu-evidence.svg', renderRates(groups, false)],
    ['assets/doppler-webgpu-evidence-mobile.svg', renderRates(groups, true)],
    ['assets/readme/generation-time.svg', renderLatency(false)],
    ['assets/readme/generation-time-mobile.svg', renderLatency(true)],
    ['assets/readme/retrieval-benchmarks.svg', renderHighlights(false)],
    ['assets/readme/retrieval-benchmarks-mobile.svg', renderHighlights(true)],
  ];
  for (const [relativePath, expected] of outputs) {
    const output = path.join(REPO_ROOT, relativePath);
    if (process.argv.includes('--check')) {
      if (fs.readFileSync(output, 'utf8') !== expected) throw new Error(`${relativePath} is stale; run npm run readme:evidence:sync`);
    } else {
      fs.writeFileSync(output, expected);
      console.log(`wrote ${relativePath}`);
    }
  }
  if (process.argv.includes('--check')) console.log('README evidence SVGs are current.');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
