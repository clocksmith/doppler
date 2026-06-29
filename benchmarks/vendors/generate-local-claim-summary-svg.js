#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  SVG_FONTS,
  SVG_THEME,
} from './svg-theme.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_LANE_ID = 'gemma-3-270m-it-q4k-rdrr';
const DEFAULT_OUTPUT = path.join(__dirname, 'results', 'doppler-vulkan-decode-grid-20260627.svg');
const CLAIM_MATRIX_PATH = path.join(__dirname, 'local-inference-claim-matrix.json');
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const WIDTH = 1200;
const HEIGHT = 850;
const PALETTE = SVG_THEME.palette;
const FONT_UI = SVG_FONTS.uiCss.replaceAll('"', "'");
const FONT_MONO = SVG_FONTS.monoCss.replaceAll('"', "'");
const BACKEND_LABELS = Object.freeze({
  'chromium-webgpu': 'Chromium',
  'node-webgpu': 'Node',
  'bun-webgpu': 'Bun',
});
const WORKLOAD_LABELS = Object.freeze({
  'p064-d064-t0-k1': 'p064 / d064',
  'p256-d128-t0-k1': 'p256 / d128',
  'p512-d128-t0-k1': 'p512 / d128',
});

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function parseArgs(argv) {
  const parsed = {
    laneId: DEFAULT_LANE_ID,
    output: DEFAULT_OUTPUT,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--lane') {
      parsed.laneId = argv[i + 1] || parsed.laneId;
      i += 1;
      continue;
    }
    if (arg.startsWith('--lane=')) {
      parsed.laneId = arg.slice('--lane='.length);
      continue;
    }
    if (arg === '--output') {
      parsed.output = argv[i + 1] || parsed.output;
      i += 1;
      continue;
    }
    if (arg.startsWith('--output=')) {
      parsed.output = arg.slice('--output='.length);
      continue;
    }
    if (arg === '--help') {
      console.log('Usage: node benchmarks/vendors/generate-local-claim-summary-svg.js [--lane <id>] [--output <svg>]');
      process.exit(0);
    }
    throw new Error(`unknown argument: ${arg}`);
  }
  return parsed;
}

function escapeXml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatNumber(value, digits = 1) {
  const number = finiteNumber(value);
  if (number == null) return 'n/a';
  if (Number.isInteger(number)) return String(number);
  return number.toFixed(digits);
}

function formatPercent(value) {
  const number = finiteNumber(value);
  if (number == null) return 'n/a';
  return `${number >= 0 ? '+' : ''}${number.toFixed(1)}%`;
}

function compareSection(report) {
  const sectionId = typeof report.decodeProfile === 'string' ? report.decodeProfile : 'throughput';
  return report?.sections?.compute?.[sectionId] || report?.sections?.compute?.throughput || null;
}

function readEngineTiming(section, engineId, metricId) {
  if (!section) return null;
  if (engineId === 'doppler') {
    return finiteNumber(section?.doppler?.result?.timing?.[metricId]);
  }
  return finiteNumber(section?.transformersjs?.timing?.[metricId] ?? section?.transformersjs?.metrics?.[metricId]);
}

function readBottleneck(section) {
  const dominant = section?.dopplerBottleneck?.dominant || {};
  return typeof dominant.label === 'string' && dominant.label.trim()
    ? dominant.label.trim()
    : typeof dominant.id === 'string' && dominant.id.trim()
      ? dominant.id.trim()
      : 'not captured';
}

function loadLaneRows(laneId) {
  const matrix = readJson(CLAIM_MATRIX_PATH);
  const lane = (Array.isArray(matrix.lanes) ? matrix.lanes : []).find((entry) => entry.id === laneId);
  if (!lane) throw new Error(`unknown local claim lane: ${laneId}`);
  const evidence = Array.isArray(lane?.evidence?.workloadCompareResults)
    ? lane.evidence.workloadCompareResults
    : [];
  if (evidence.length === 0) throw new Error(`lane ${laneId} has no workloadCompareResults`);

  return evidence.map((entry) => {
    const comparePath = path.join(REPO_ROOT, ...entry.compareResult.split('/'));
    const report = readJson(comparePath);
    const section = compareSection(report);
    const dopplerDecode = readEngineTiming(section, 'doppler', 'decodeTokensPerSec');
    const tjsDecode = readEngineTiming(section, 'transformersjs', 'decodeTokensPerSec');
    const dopplerPrefill = readEngineTiming(section, 'doppler', 'prefillTokensPerSec');
    const tjsPrefill = readEngineTiming(section, 'transformersjs', 'prefillTokensPerSec');
    const dopplerLeads = dopplerDecode != null && tjsDecode != null && dopplerDecode >= tjsDecode;
    const leader = dopplerDecode == null || tjsDecode == null
      ? 'unknown'
      : dopplerLeads ? 'doppler' : 'tjs';
    const leaderBase = leader === 'doppler' ? tjsDecode : dopplerDecode;
    const leaderValue = leader === 'doppler' ? dopplerDecode : tjsDecode;
    const leadPercent = leaderBase && leaderValue
      ? ((leaderValue - leaderBase) / leaderBase) * 100
      : null;
    return {
      backendId: entry.backendId,
      workloadId: entry.workloadId,
      compareResult: entry.compareResult,
      dopplerDecode,
      tjsDecode,
      dopplerPrefill,
      tjsPrefill,
      leader,
      leadPercent,
      bottleneck: readBottleneck(section),
      exactMatch: report?.correctness?.exactMatch === true,
      runs: finiteNumber(report.runs),
    };
  });
}

function rect(x, y, width, height, fill, extra = '') {
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="5" fill="${fill}"${extra} />`;
}

function text(x, y, value, options = {}) {
  const fill = options.fill || PALETTE.text;
  const size = options.size || 13;
  const weight = options.weight ? ` font-weight="${options.weight}"` : '';
  const anchor = options.anchor ? ` text-anchor="${options.anchor}"` : '';
  const family = options.mono ? FONT_MONO : FONT_UI;
  return `<text x="${x}" y="${y}" fill="${fill}" font-family="${family}" font-size="${size}"${weight}${anchor}>${escapeXml(value)}</text>`;
}

function renderCell(row, x, y, width, height, maxDecode) {
  const leaderColor = row.leader === 'doppler' ? PALETTE.doppler : PALETTE.bad;
  const fill = row.leader === 'doppler' ? PALETTE.leaderDoppler : PALETTE.leaderTjs;
  const stroke = row.leader === 'doppler' ? PALETTE.doppler : PALETTE.bad;
  const barX = x + 22;
  const barWidth = width - 44;
  const dopplerWidth = Math.max(2, (row.dopplerDecode / maxDecode) * barWidth);
  const tjsWidth = Math.max(2, (row.tjsDecode / maxDecode) * barWidth);
  const leaderText = row.leader === 'doppler'
    ? `Doppler ${formatPercent(row.leadPercent)}`
    : `TJS ${formatPercent(row.leadPercent)}`;
  const exact = row.exactMatch ? 'exact' : 'check';
  return [
    rect(x, y, width, height, fill, ` stroke="${stroke}" stroke-width="1.25"`),
    text(x + 18, y + 28, BACKEND_LABELS[row.backendId] || row.backendId, { size: 18, weight: 800 }),
    text(x + width - 18, y + 28, leaderText, { size: 13, weight: 800, anchor: 'end', fill: leaderColor }),
    rect(barX, y + 48, dopplerWidth, 20, PALETTE.doppler),
    text(barX + 8, y + 63, `Doppler ${formatNumber(row.dopplerDecode)} tok/s`, {
      size: 12,
      weight: 800,
      fill: PALETTE.bg,
      mono: true,
    }),
    rect(barX, y + 78, tjsWidth, 20, PALETTE.transformersjs),
    text(barX + 8, y + 93, `TJS ${formatNumber(row.tjsDecode)} tok/s`, {
      size: 12,
      weight: 800,
      fill: PALETTE.text,
      mono: true,
    }),
    text(x + 18, y + 122, `prefill ${formatNumber(row.dopplerPrefill)} / ${formatNumber(row.tjsPrefill)} tok/s`, {
      size: 12,
      fill: PALETTE.muted,
      mono: true,
    }),
    text(x + 18, y + 143, `${exact} • ${formatNumber(row.runs, 0)} runs • ${row.bottleneck}`, {
      size: 12,
      fill: PALETTE.muted,
    }),
  ].join('\n');
}

function renderSvg(rows) {
  const workloads = [...new Set(rows.map((row) => row.workloadId))];
  const backends = ['chromium-webgpu', 'node-webgpu', 'bun-webgpu'];
  const rowsByKey = new Map(rows.map((row) => [`${row.backendId}:${row.workloadId}`, row]));
  const maxDecode = Math.max(1, ...rows.flatMap((row) => [row.dopplerDecode || 0, row.tjsDecode || 0]));
  const cellWidth = 330;
  const cellHeight = 160;
  const gap = 18;
  const left = 150;
  const top = 198;
  const dopplerLeading = rows.filter((row) => row.leader === 'doppler').length;
  const tjsLeading = rows.filter((row) => row.leader === 'tjs').length;

  const body = [];
  body.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-labelledby="title desc">`);
  body.push(`<title id="title">Doppler AMD Vulkan decode throughput grid</title>`);
  body.push(`<desc id="desc">Gemma 3 270M Q4K throughput receipts on AMD Vulkan across p064, p256, p512 and Chromium, Node, Bun, with Doppler-leading and TJS-leading cells marked.</desc>`);
  body.push('<defs><style>');
  body.push(`text{font-family:${FONT_UI};letter-spacing:0}.mono{font-family:${FONT_MONO}}`);
  body.push('</style></defs>');
  body.push(rect(0, 0, WIDTH, HEIGHT, PALETTE.bg, ''));
  body.push(rect(18, 18, WIDTH - 36, HEIGHT - 36, PALETTE.bg, ` stroke="${PALETTE.border}" stroke-width="1.5"`));
  body.push(text(42, 58, 'AMD VULKAN DECODE GRID', { size: 12, weight: 800 }));
  body.push(text(42, 96, 'Doppler / Transformers.js decode throughput', { size: 34, weight: 850 }));
  body.push(text(42, 126, 'Gemma 3 270M IT Q4K - greedy - warm cache - 15 timed runs - exact output match - no hidden fallback', { size: 15, fill: PALETTE.muted }));
  body.push(rect(900, 50, 230, 58, PALETTE.text));
  body.push(text(1015, 73, `${dopplerLeading} Doppler-leading`, { size: 16, weight: 850, fill: PALETTE.bg, anchor: 'middle' }));
  body.push(text(1015, 96, `${tjsLeading} TJS-leading`, { size: 13, weight: 800, fill: PALETTE.bg, anchor: 'middle' }));

  for (const [index, backendId] of backends.entries()) {
    body.push(text(left + index * (cellWidth + gap) + cellWidth / 2, 166, BACKEND_LABELS[backendId] || backendId, {
      size: 14,
      weight: 800,
      anchor: 'middle',
      fill: PALETTE.muted,
    }));
  }

  for (const [rowIndex, workloadId] of workloads.entries()) {
    const y = top + rowIndex * (cellHeight + gap);
    body.push(text(42, y + 74, WORKLOAD_LABELS[workloadId] || workloadId, { size: 18, weight: 850 }));
    body.push(text(42, y + 96, workloadId, { size: 11, fill: PALETTE.muted, mono: true }));
    for (const [columnIndex, backendId] of backends.entries()) {
      const row = rowsByKey.get(`${backendId}:${workloadId}`);
      if (!row) continue;
      const x = left + columnIndex * (cellWidth + gap);
      body.push(renderCell(row, x, y, cellWidth, cellHeight, maxDecode));
    }
  }

  body.push(`<line x1="42" y1="770" x2="1158" y2="770" stroke="${PALETTE.grid}" />`);
  body.push(text(42, 798, 'Read as decode throughput, higher is better. Blue cells are Doppler-leading; red cells are TJS-leading.', {
    size: 13,
    fill: PALETTE.muted,
  }));
  body.push(text(42, 820, 'Receipts: compare_20260627T202549..203348 plus p512 200323/200603/200811. Broader medium-model claim remains gated.', {
    size: 12,
    fill: PALETTE.muted,
    mono: true,
  }));
  body.push('</svg>');
  return body.join('\n');
}

function main() {
  const options = parseArgs(process.argv);
  const rows = loadLaneRows(options.laneId);
  const outputPath = path.resolve(options.output);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${renderSvg(rows)}\n`, 'utf8');
  console.log(`wrote ${outputPath}`);
}

const isDirectRun = process.argv[1]
  && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isDirectRun) {
  try {
    main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}
