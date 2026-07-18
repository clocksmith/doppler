#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT_PATH = path.join(REPO_ROOT, 'assets', 'doppler-webgpu-evidence.svg');
const RECEIPT_BASE_URL = 'https://github.com/clocksmith/doppler/blob/main/';

const WIDTH = 1200;
const MARGIN = 30;
const MODEL_X = 30;
const VALUE_X = 300;
const PLOT_START_X = 575;
const PLOT_END_X = 955;
const DIVIDER_X = 995;
const LEAD_X = 1090;
const SECTION_HEIGHT = 46;
const ROW_HEIGHT = 88;

const groups = [
  {
    label: 'Apple M3 / Metal',
    rows: [
      {
        model: 'Qwen 3.5 0.8B text',
        kind: 'text',
        unit: 'tok/s',
        receipt: 'benchmarks/vendors/results/compare_20260709T154633.json',
      },
      {
        model: 'Qwen 3 Embedding 0.6B',
        kind: 'embedding',
        unit: 'embedding/s',
        receipt: 'benchmarks/vendors/results/embedding_compare_qwen-3-embedding-0-6b-q4k-ehf16-af32_20260709T180853.json',
      },
      {
        model: 'Qwen 3 Reranker 0.6B',
        kind: 'rerank',
        unit: 'rerank/s',
        receipt: 'benchmarks/vendors/results/rerank_compare_qwen-3-reranker-0-6b-q4k-ehf16-af32_20260709T192830.json',
      },
      {
        model: 'Qwen 3.5 2B text',
        kind: 'paired-text',
        unit: 'tok/s',
        localArtifact: true,
        receipt: 'benchmarks/vendors/results/qwen35-2b-metal-paired-p256-d512-20260710.json',
      },
    ],
  },
  {
    label: 'Radeon 8060S / Vulkan',
    rows: [
      {
        model: 'Qwen 3.5 0.8B text',
        kind: 'text',
        unit: 'tok/s',
        receipt: 'benchmarks/vendors/results/compare_20260707T153509.json',
      },
      {
        model: 'Qwen 3 Embedding 0.6B',
        kind: 'embedding',
        unit: 'embedding/s',
        receipt: 'benchmarks/vendors/results/embedding_compare_qwen-3-embedding-0-6b-q4k-ehf16-af32_20260710T011455.json',
      },
      {
        model: 'Qwen 3 Reranker 0.6B',
        kind: 'rerank',
        unit: 'rerank/s',
        receipt: 'benchmarks/vendors/results/rerank_compare_qwen-3-reranker-0-6b-q4k-ehf16-af32_20260710T014450.json',
      },
      {
        model: 'Qwen 3.5 2B text',
        kind: 'text',
        unit: 'tok/s',
        localArtifact: true,
        receipt: 'benchmarks/vendors/results/compare_20260707T161623.json',
      },
      {
        model: 'Gemma 4 E2B INT4-PLE',
        kind: 'text',
        unit: 'tok/s',
        localArtifact: true,
        receipt: 'benchmarks/vendors/results/compare_20260707T170557.json',
      },
    ],
  },
];

function requireFinite(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`${label} must be finite`);
  }
  return number;
}

function sampleStats(values, label) {
  const samples = values.map((value, index) => requireFinite(value, `${label}[${index}]`));
  if (samples.length < 2) {
    throw new Error(`${label} must contain at least two samples`);
  }
  const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  const variance = samples.reduce((sum, value) => sum + ((value - mean) ** 2), 0)
    / (samples.length - 1);
  const stdDev = Math.sqrt(variance);
  const min = Math.min(...samples);
  const max = Math.max(...samples);
  return {
    mean,
    min,
    max,
    bandMin: Math.max(min, mean - stdDev),
    bandMax: Math.min(max, mean + stdDev),
    samples: samples.length,
  };
}

function rateStats(stats, label) {
  const mean = requireFinite(stats?.mean, `${label}.mean`);
  const stdDev = requireFinite(stats?.stdDev, `${label}.stdDev`);
  const min = requireFinite(stats?.min, `${label}.min`);
  const max = requireFinite(stats?.max, `${label}.max`);
  const samples = requireFinite(stats?.samples, `${label}.samples`);
  return {
    mean,
    min,
    max,
    bandMin: Math.max(min, mean - stdDev),
    bandMax: Math.min(max, mean + stdDev),
    samples,
  };
}

function reciprocalLatencyStats(stats, label) {
  const meanMs = requireFinite(stats?.mean, `${label}.mean`);
  const stdDevMs = requireFinite(stats?.stdDev, `${label}.stdDev`);
  const minMs = requireFinite(stats?.min, `${label}.min`);
  const maxMs = requireFinite(stats?.max, `${label}.max`);
  const samples = requireFinite(stats?.samples, `${label}.samples`);
  if (minMs <= 0 || meanMs - stdDevMs <= 0) {
    throw new Error(`${label} must be positive after its one-SD lower bound`);
  }
  const min = 1000 / maxMs;
  const max = 1000 / minMs;
  return {
    mean: 1000 / meanMs,
    min,
    max,
    bandMin: Math.max(min, 1000 / (meanMs + stdDevMs)),
    bandMax: Math.min(max, 1000 / (meanMs - stdDevMs)),
    samples,
  };
}

function textStats(receipt, receiptPath) {
  const parity = receipt?.sections?.compute?.parity;
  const doppler = rateStats(
    parity?.doppler?.result?.metrics?.throughput?.decodeTokensPerSec,
    `${receiptPath}: Doppler decode throughput`
  );
  const transformersjs = sampleStats(
    (parity?.transformersjs?.runs ?? []).map((run) => run?.decodeTokensPerSec),
    `${receiptPath}: Transformers.js decode throughput`
  );
  return { doppler, transformersjs };
}

function pairedTextStats(receipt, receiptPath) {
  const pairs = Array.isArray(receipt?.pairs) ? receipt.pairs : [];
  return {
    doppler: sampleStats(
      pairs.map((pair) => pair?.dopplerTokensPerSec),
      `${receiptPath}: Doppler paired decode throughput`
    ),
    transformersjs: sampleStats(
      pairs.map((pair) => pair?.transformersjsTokensPerSec),
      `${receiptPath}: Transformers.js paired decode throughput`
    ),
  };
}

function retrievalStats(receipt, receiptPath, kind) {
  const isEmbedding = kind === 'embedding';
  const metricName = isEmbedding ? 'timedEmbeddingMs' : 'timedRerankMs';
  const runMetricName = isEmbedding ? 'embeddingMs' : 'rerankMs';
  const dopplerLatency = receipt?.raw?.dopplerBench?.metrics?.latency?.[metricName];
  const transformersjsRuns = receipt?.raw?.transformersjsBench?.runs ?? [];
  return {
    doppler: reciprocalLatencyStats(
      dopplerLatency,
      `${receiptPath}: Doppler ${kind} latency`
    ),
    transformersjs: sampleStats(
      transformersjsRuns.map((run) => 1000 / requireFinite(
        run?.[runMetricName],
        `${receiptPath}: Transformers.js ${runMetricName}`
      )),
      `${receiptPath}: Transformers.js ${kind} throughput`
    ),
  };
}

async function loadRow(row) {
  const receiptPath = path.join(REPO_ROOT, row.receipt);
  const receipt = JSON.parse(await fs.readFile(receiptPath, 'utf8'));
  let stats;
  if (row.kind === 'text') {
    stats = textStats(receipt, row.receipt);
  } else if (row.kind === 'paired-text') {
    stats = pairedTextStats(receipt, row.receipt);
  } else {
    stats = retrievalStats(receipt, row.receipt, row.kind);
  }
  if (stats.doppler.samples !== stats.transformersjs.samples) {
    throw new Error(`${row.receipt}: engine sample counts must match`);
  }
  if (stats.doppler.mean <= stats.transformersjs.mean) {
    throw new Error(`${row.receipt}: README win claim no longer holds`);
  }
  return {
    ...row,
    ...stats,
    samples: stats.doppler.samples,
    lead: stats.doppler.mean / stats.transformersjs.mean,
  };
}

function escapeXml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function formatRate(value) {
  if (value >= 10) return value.toFixed(2);
  return value.toFixed(2);
}

function formatAxis(value) {
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function scaleForRow(row) {
  const observedMin = Math.min(row.doppler.min, row.transformersjs.min);
  const observedMax = Math.max(row.doppler.max, row.transformersjs.max);
  const span = observedMax - observedMin;
  const padding = span > 0 ? span * 0.08 : Math.max(observedMax * 0.08, 1);
  const min = Math.max(0, observedMin - padding);
  const max = observedMax + padding;
  return {
    min,
    max,
    x(value) {
      return PLOT_START_X + ((value - min) / (max - min)) * (PLOT_END_X - PLOT_START_X);
    },
  };
}

function distribution(stats, y, prefix, scale) {
  const minX = scale.x(stats.min);
  const maxX = scale.x(stats.max);
  const bandMinX = scale.x(stats.bandMin);
  const bandMaxX = scale.x(stats.bandMax);
  const meanX = scale.x(stats.mean);
  const bandWidth = Math.max(1, bandMaxX - bandMinX);
  return [
    `<path d="M${PLOT_START_X} ${y}H${PLOT_END_X}" class="axis"/>`,
    `<path d="M${minX.toFixed(1)} ${y}H${maxX.toFixed(1)} M${minX.toFixed(1)} ${y - 9}V${y + 9} M${maxX.toFixed(1)} ${y - 9}V${y + 9}" class="${prefix}-mark"/>`,
    `<rect x="${bandMinX.toFixed(1)}" y="${y - 10}" width="${bandWidth.toFixed(1)}" height="20" rx="2" class="${prefix}-box"/>`,
    `<circle cx="${meanX.toFixed(1)}" cy="${y}" r="5" class="${prefix}-dot"/>`,
  ].join('');
}

function renderRow(row, y) {
  const scale = scaleForRow(row);
  const detail = row.localArtifact
    ? `n=${row.samples} / local artifact`
    : `n=${row.samples}`;
  const receiptUrl = `${RECEIPT_BASE_URL}${row.receipt}`;
  const tooltip = [
    `${row.model}.`,
    `Doppler ${formatRate(row.doppler.mean)} ${row.unit}, range ${formatRate(row.doppler.min)}-${formatRate(row.doppler.max)}.`,
    `Transformers.js ${formatRate(row.transformersjs.mean)} ${row.unit}, range ${formatRate(row.transformersjs.min)}-${formatRate(row.transformersjs.max)}.`,
  ].join(' ');
  return [
    `<g class="benchmark-row" data-receipt="${escapeXml(row.receipt)}">`,
    `<title>${escapeXml(tooltip)}</title>`,
    `<a href="${escapeXml(receiptUrl)}" target="_blank">`,
    `<text x="${MODEL_X}" y="${y + 25}" class="model receipt">${escapeXml(row.model)}</text>`,
    `</a>`,
    `<text x="${MODEL_X}" y="${y + 56}" class="detail">${escapeXml(detail)}</text>`,
    `<text x="${VALUE_X}" y="${y + 25}" class="dv mono">${formatRate(row.doppler.mean)} ${escapeXml(row.unit)}</text>`,
    `<text x="${VALUE_X}" y="${y + 58}" class="tv mono">${formatRate(row.transformersjs.mean)} ${escapeXml(row.unit)}</text>`,
    distribution(row.doppler, y + 20, 'd', scale),
    distribution(row.transformersjs, y + 53, 't', scale),
    `<text x="${PLOT_START_X}" y="${y + 80}" class="scale mono">${formatAxis(scale.min)}</text>`,
    `<text x="${PLOT_END_X}" y="${y + 80}" text-anchor="end" class="scale mono">${formatAxis(scale.max)}</text>`,
    `<path d="M${DIVIDER_X} ${y + 4}V${y + 72}" class="divider"/>`,
    `<text x="${LEAD_X}" y="${y + 49}" text-anchor="middle" class="ratio">${row.lead.toFixed(2)}x</text>`,
    `<path d="M${MARGIN} ${y + ROW_HEIGHT - 1}H${WIDTH - MARGIN}" class="row"/>`,
    `</g>`,
  ].join('\n');
}

async function renderSvg() {
  const loadedGroups = [];
  for (const group of groups) {
    loadedGroups.push({
      ...group,
      rows: await Promise.all(group.rows.map(loadRow)),
    });
  }

  const body = [];
  let y = 110;
  for (let index = 0; index < loadedGroups.length; index += 1) {
    const group = loadedGroups[index];
    body.push(`<rect x="${MARGIN}" y="${y}" width="${WIDTH - (2 * MARGIN)}" height="${SECTION_HEIGHT}" class="band"/>`);
    body.push(`<text x="${MARGIN + 14}" y="${y + 31}" class="section">${escapeXml(group.label)}</text>`);
    body.push(`<text x="${WIDTH - MARGIN - 14}" y="${y + 30}" text-anchor="end" class="section-detail">${group.rows.length} MODELS</text>`);
    y += SECTION_HEIGHT + 10;
    for (const row of group.rows) {
      body.push(renderRow(row, y));
      y += ROW_HEIGHT;
    }
    if (index < loadedGroups.length - 1) y += 22;
  }

  const footerLineY = y + 20;
  const height = footerLineY + 82;
  return `<!-- Generated by tools/render-readme-evidence-chart.js. -->
<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${height}" viewBox="0 0 ${WIDTH} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">Doppler and Transformers.js WebGPU throughput</title>
  <desc id="desc">Mean throughput, one-standard-deviation bands, and observed ranges for nine browser WebGPU comparisons on Apple M3 Metal and Radeon 8060S Vulkan. Higher throughput is farther right. Retrieval bands are reciprocal transforms of receipt latency statistics.</desc>
  <defs><style>
    text{fill:#111;font-family:Inter,"Segoe UI","Helvetica Neue",Arial,sans-serif;letter-spacing:0}.mono{font-family:SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}.legend{font-size:16px;font-weight:750;fill:#495057}.column{font-size:14px;font-weight:800;fill:#6b7280}.section{font-size:22px;font-weight:800}.section-detail{font-size:14px;font-weight:750;fill:#495057}.model{font-size:20px;font-weight:800}.detail{font-size:15px;font-weight:650;fill:#6b7280}.dv,.tv{font-size:17px;font-weight:800}.dv{fill:#6d28d9}.tv{fill:#9a6700}.ratio{font-size:28px;font-weight:800;fill:#6d28d9}.footer{font-size:14px;font-weight:650;fill:#495057}.scale{font-size:11px;font-weight:650;fill:#6b7280}.band{fill:#f4f4f5}.row{stroke:#e5e7eb}.axis{stroke:#e5e7eb;stroke-width:3}.divider{stroke:#e5e7eb}.d-box{fill:#ede9fe;stroke:#6d28d9;stroke-width:3}.t-box{fill:#fff4cc;stroke:#ffb000;stroke-width:3}.d-mark{fill:none;stroke:#6d28d9;stroke-width:3}.t-mark{fill:none;stroke:#ffb000;stroke-width:3}.d-dot{fill:#6d28d9}.t-dot{fill:#ffb000}.receipt:hover{text-decoration:underline}
  </style></defs>
  <rect width="${WIDTH}" height="${height}" fill="#fff"/>

  <circle cx="39" cy="34" r="8" fill="#6d28d9"/><text x="56" y="40" class="legend">Doppler</text>
  <circle cx="162" cy="34" r="8" fill="#ffb000"/><text x="179" y="40" class="legend">Transformers.js</text>
  <text x="${WIDTH - MARGIN}" y="40" text-anchor="end" class="legend">BAND +/- 1 SD / DOT MEAN / WHISKER MIN-MAX</text>
  <path d="M${MARGIN} 64H${WIDTH - MARGIN}" stroke="#111"/>
  <text x="${MODEL_X}" y="92" class="column">MODEL</text>
  <text x="${VALUE_X}" y="92" class="column">MEAN RATE</text>
  <text x="${PLOT_START_X}" y="92" class="column">RATE DISTRIBUTION / FASTER &gt;</text>
  <text x="${LEAD_X}" y="92" text-anchor="middle" class="column">LEAD</text>

  ${body.join('\n  ')}

  <path d="M${MARGIN} ${footerLineY}H${WIDTH - MARGIN}" stroke="#111"/>
  <text x="${MARGIN}" y="${footerLineY + 29}" class="footer">Per-row throughput scale / higher is right / band +/- 1 SD / whisker min-max.</text>
  <text x="${MARGIN}" y="${footerLineY + 56}" class="footer">Retrieval bands invert receipt latency statistics; model names open the source receipt.</text>
  <text x="${WIDTH - MARGIN}" y="${footerLineY + 56}" text-anchor="end" class="footer mono">9 RESULTS</text>
</svg>
`;
}

async function main() {
  const args = process.argv.slice(2);
  const check = args.includes('--check');
  const unknown = args.filter((arg) => arg !== '--check');
  if (unknown.length > 0) {
    throw new Error(`Unknown argument: ${unknown[0]}`);
  }
  const rendered = await renderSvg();
  if (check) {
    const current = await fs.readFile(OUTPUT_PATH, 'utf8');
    if (current !== rendered) {
      throw new Error('README evidence chart is stale; run npm run readme:evidence:sync');
    }
    console.log('README evidence chart is up to date');
    return;
  }
  await fs.writeFile(OUTPUT_PATH, rendered, 'utf8');
  console.log(`Wrote ${path.relative(REPO_ROOT, OUTPUT_PATH)}`);
}

await main();
