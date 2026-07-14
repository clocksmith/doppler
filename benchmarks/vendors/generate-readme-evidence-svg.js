#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const OUTPUT_PATH = path.join(REPO_ROOT, 'assets', 'doppler-webgpu-evidence.svg');
const WIDTH = 1280;
const HEIGHT = 930;
const AXIS_START = 600;
const AXIS_END = 970;
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
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`missing finite ${label}`);
  return number;
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

function loadRow(definition) {
  const report = readJson(definition.receipt);
  if (definition.kind === 'text') return loadTextRow(definition, report);
  if (definition.kind === 'paired-text') return loadPairedTextRow(definition, report);
  if (definition.kind === 'embedding' || definition.kind === 'rerank') {
    return loadRetrievalRow(definition, report);
  }
  throw new Error(`unsupported evidence row kind: ${definition.kind}`);
}

function escapeXml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function formatRate(value, unit) {
  return `${RATE_FORMATTER.format(value)} ${unit}`;
}

function scaleForRow(row) {
  const values = [
    row.doppler.min,
    row.doppler.max,
    row.doppler.rate,
    row.transformersjs.min,
    row.transformersjs.max,
    row.transformersjs.rate,
  ];
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const span = Math.max(maximum - minimum, maximum * 0.02, 0.01);
  const low = minimum - span * 0.08;
  const high = maximum + span * 0.08;
  return (value) => AXIS_START + ((value - low) / (high - low)) * (AXIS_END - AXIS_START);
}

function renderRange(scale, y, values, prefix) {
  const minimum = scale(values.min);
  const maximum = scale(values.max);
  const rate = scale(values.rate);
  const width = Math.max(2, maximum - minimum);
  return [
    `<path d="M${minimum.toFixed(1)} ${y}H${maximum.toFixed(1)}" class="${prefix}-mark"/>`,
    `<rect x="${minimum.toFixed(1)}" y="${y - 7}" width="${width.toFixed(1)}" height="14" class="${prefix}-box"/>`,
    `<path d="M${minimum.toFixed(1)} ${y - 8}V${y + 8}M${maximum.toFixed(1)} ${y - 8}V${y + 8}M${rate.toFixed(1)} ${y - 9}V${y + 9}" class="${prefix}-mark"/>`,
  ].join('');
}

function renderRow(row, top) {
  const scale = scaleForRow(row);
  const detail = [`n=${row.runs}`];
  if (row.detail) detail.push(row.detail);
  const ratio = row.doppler.rate / row.transformersjs.rate;
  return [
    `<text x="42" y="${top + 24}" class="model">${escapeXml(row.model)}</text>`,
    `<text x="42" y="${top + 49}" class="detail">${escapeXml(detail.join(' · '))}</text>`,
    `<text x="320" y="${top + 23}" class="dv mono">${formatRate(row.doppler.rate, row.unit)}</text>`,
    `<text x="320" y="${top + 49}" class="tv mono">${formatRate(row.transformersjs.rate, row.unit)}</text>`,
    `<path d="M${AXIS_START} ${top + 19}H${AXIS_END}" class="axis"/>`,
    renderRange(scale, top + 19, row.doppler, 'd'),
    `<path d="M${AXIS_START} ${top + 45}H${AXIS_END}" class="axis"/>`,
    renderRange(scale, top + 45, row.transformersjs, 't'),
    `<path d="M1010 ${top + 8}V${top + 56}" class="divider"/>`,
    `<text x="1109" y="${top + 41}" text-anchor="middle" class="ratio">${ratio.toFixed(2)}x</text>`,
    `<path d="M42 ${top + 72}H1238" class="row"/>`,
  ].join('\n  ');
}

function renderSvg(groups) {
  const paired = groups.flatMap((group) => group.rows).find((row) => row.kind === 'paired-text');
  const body = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-labelledby="title desc">`,
    '  <title id="title">Doppler and Transformers.js browser WebGPU throughput</title>',
    '  <desc id="desc">Reported throughput and receipt-reported min-to-max rate ranges for nine browser WebGPU comparisons on Apple M3 Metal and Radeon 8060S Vulkan. Higher is faster.</desc>',
    '  <defs><style>',
    '    text{fill:#111;font-family:Inter,"Segoe UI","Helvetica Neue",Arial,sans-serif;letter-spacing:0}.mono{font-family:SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}.legend{font-size:12px;font-weight:750;fill:#495057}.column{font-size:11px;font-weight:800;fill:#6b7280}.section{font-size:18px;font-weight:800}.section-detail{font-size:12px;font-weight:750;fill:#495057}.model{font-size:16px;font-weight:800}.detail{font-size:12px;font-weight:650;fill:#6b7280}.dv,.tv{font-size:13px;font-weight:800}.dv{fill:#6d28d9}.tv{fill:#9a6700}.ratio{font-size:24px;font-weight:800;fill:#6d28d9}.footer{font-size:11px;font-weight:650;fill:#495057}.band{fill:#f4f4f5}.row{stroke:#e5e7eb}.axis{stroke:#e5e7eb;stroke-width:2}.divider{stroke:#e5e7eb}.d-box{fill:#ede9fe;stroke:#6d28d9;stroke-width:2}.t-box{fill:#fff4cc;stroke:#ffb000;stroke-width:2}.d-mark{fill:none;stroke:#6d28d9;stroke-width:2}.t-mark{fill:none;stroke:#ffb000;stroke-width:2}',
    '  </style></defs>',
    `  <rect width="${WIDTH}" height="${HEIGHT}" fill="#fff"/>`,
    '',
    '  <circle cx="49" cy="32" r="7" fill="#6d28d9"/><text x="65" y="37" class="legend">Doppler</text>',
    '  <circle cx="149" cy="32" r="7" fill="#ffb000"/><text x="165" y="37" class="legend">Transformers.js</text>',
    '  <text x="1238" y="37" text-anchor="end" class="legend">BAND MIN–MAX · CENTER REPORTED RATE</text>',
    '  <path d="M42 56H1238" stroke="#111"/>',
    '  <text x="42" y="82" class="column">MODEL</text><text x="320" y="82" class="column">THROUGHPUT</text>',
    '  <text x="600" y="82" class="column">RECEIPT RATE RANGE · HIGHER IS FASTER →</text><text x="1109" y="82" text-anchor="middle" class="column">RATE RATIO</text>',
  ];

  let sectionTop = 94;
  for (const [groupIndex, group] of groups.entries()) {
    body.push('');
    body.push(`  <rect x="42" y="${sectionTop}" width="1196" height="38" class="band"/>`);
    body.push(`  <text x="54" y="${sectionTop + 26}" class="section">${escapeXml(group.label)}</text><text x="1226" y="${sectionTop + 25}" text-anchor="end" class="section-detail">${group.rows.length} MODELS</text>`);
    let rowTop = sectionTop + 46;
    for (const row of group.rows) {
      body.push('');
      body.push(`  ${renderRow(row, rowTop)}`);
      rowTop += 72;
    }
    sectionTop = rowTop + (groupIndex === 0 ? 12 : 0);
  }

  body.push('');
  body.push('  <path d="M42 858H1238" stroke="#111"/>');
  body.push('  <text x="42" y="883" class="footer">Each row uses its own throughput scale; faster is right.</text>');
  body.push(`  <text x="42" y="906" class="footer">Metal Qwen 3.5 2B: ${paired.runs} paired runs · ${paired.pairWins.doppler} Doppler wins · ${paired.pairWins.transformersjs} TJS wins · exact tokens.</text>`);
  body.push('  <text x="1238" y="906" text-anchor="end" class="footer mono">9 LINKED RESULTS</text>');
  body.push('</svg>');
  return `${body.join('\n')}\n`;
}

function main() {
  const unknownArgs = process.argv.slice(2).filter((arg) => arg !== '--check');
  if (unknownArgs.length > 0) throw new Error(`unknown argument: ${unknownArgs[0]}`);
  const groups = GROUPS.map((group) => ({
    ...group,
    rows: group.rows.map(loadRow),
  }));
  const expected = renderSvg(groups);
  if (process.argv.includes('--check')) {
    const actual = fs.readFileSync(OUTPUT_PATH, 'utf8');
    if (actual !== expected) throw new Error('assets/doppler-webgpu-evidence.svg is stale; run npm run readme:evidence:sync');
    console.log('README evidence SVG is current.');
    return;
  }
  fs.writeFileSync(OUTPUT_PATH, expected, 'utf8');
  console.log(`wrote ${OUTPUT_PATH}`);
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
