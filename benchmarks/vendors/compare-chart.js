#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

const CHART_TYPES = Object.freeze(['bar', 'stacked', 'radar', 'phases']);
const DEFAULT_CHART = 'bar';
const DEFAULT_WIDTH = 960;
const DEFAULT_HEIGHT = 560;
const DEFAULT_SECTION = 'compute/parity';

const DEFAULT_METRICS = Object.freeze([
  {
    id: 'decodeTokensPerSec',
    label: 'Decode tok/s',
    unit: 'tok/s',
    higherBetter: true,
  },
  {
    id: 'prefillTokensPerSec',
    label: 'Prompt tok/s (prefill)',
    unit: 'tok/s',
    higherBetter: true,
  },
  {
    id: 'firstTokenMs',
    label: 'TTFT (first token)',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'firstResponseMs',
    label: 'First response (load + TTFT)',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'prefillMs',
    label: 'Prefill',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'decodeMs',
    label: 'Decode',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'totalRunMs',
    label: 'Total run',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'modelLoadMs',
    label: 'Model load',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'decodeMsPerTokenP50',
    label: 'Decode p50 ms/token',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'decodeMsPerTokenP95',
    label: 'Decode p95 ms/token',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'decodeMsPerTokenP99',
    label: 'Decode p99 ms/token',
    unit: 'ms',
    higherBetter: false,
  },
]);

const FALLBACK_SECTIONS = Object.freeze([
  'compute/parity',
  'compute/throughput',
  'compute',
  'warm',
  'cold',
]);

const METRIC_PATH_HINTS = Object.freeze({
  decodeTokensPerSec: [
    'result.timing.decodeTokensPerSec',
  ],
  prefillTokensPerSec: [
    'result.timing.prefillTokensPerSec',
  ],
  firstTokenMs: [
    'result.timing.firstTokenMs',
  ],
  firstResponseMs: [
    'result.timing.firstResponseMs',
  ],
  prefillMs: [
    'result.timing.prefillMs',
  ],
  decodeMs: [
    'result.timing.decodeMs',
  ],
  totalRunMs: [
    'result.timing.totalRunMs',
  ],
  modelLoadMs: [
    'result.timing.modelLoadMs',
  ],
  decodeMsPerTokenP50: [
    'result.timing.decodeMsPerTokenP50',
  ],
  decodeMsPerTokenP95: [
    'result.timing.decodeMsPerTokenP95',
  ],
  decodeMsPerTokenP99: [
    'result.timing.decodeMsPerTokenP99',
  ],
});

const PALETTE = Object.freeze({
  bg: '#000000',
  text: '#ffffff',
  muted: '#cbd5e1',
  grid: '#1f2937',
  doppler: '#9d4edd',
  tjs: '#ffbd45',
  fail: '#3f3f46',
  failFill: '#7f1d1d',
  metric: [
    '#9d4edd',
    '#c77dff',
    '#7c3aed',
    '#ffbd45',
    '#ffd580',
    '#22d3ee',
    '#4ade80',
    '#f59e0b',
  ],
});

const ENGINE_META = Object.freeze({
  doppler: {
    key: 'doppler',
    label: 'Doppler.js',
    color: PALETTE.doppler,
  },
  tjs: {
    key: 'tjs',
    label: 'Transformers.js',
    color: PALETTE.tjs,
  },
});

const DERIVED_KEY = Object.freeze({
  doppler: 'doppler',
  tjs: 'transformersjs',
});

function usage() {
  return [
    'Usage: node benchmarks/vendors/compare-chart.js --input <file> [--input <file2> ...]',
    '',
    'Options:',
    '  --input <path>                Benchmark JSON output path (repeatable for multi-workload)',
    '  --output <path>               Output SVG path',
    '  --section <path>              Section path in result payload (default: compute/parity)',
    '  --chart <bar|stacked|radar|phases>  Chart family (default: bar)',
    '  --metrics <id,id,...>         Comma-separated metric IDs',
    '  --width <n>                   SVG width (default: 960)',
    '  --height <n>                  SVG height (default: 560)',
    '  --help                        Show this help text',
    '',
    'Examples:',
    '  node benchmarks/vendors/compare-chart.js --input benchmarks/vendors/results/compare_latest.json',
    '  node benchmarks/vendors/compare-chart.js --input ... --chart stacked --width 1200',
    '  node benchmarks/vendors/compare-chart.js --input ... --chart radar --metrics decodeTokensPerSec,firstTokenMs',
    '  node benchmarks/vendors/compare-chart.js --input ... --chart phases',
    '  node benchmarks/vendors/compare-chart.js --chart phases --input workload1.json --input workload2.json',
    '  node benchmarks/vendors/compare-chart.js --chart radar --input workload1.json --input workload2.json',
  ].join('\n');
}

function parseArgs(argv) {
  const parsed = {
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT,
    chart: DEFAULT_CHART,
    section: DEFAULT_SECTION,
    metricIds: [],
    inputs: [],
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];

    if (arg === '--help' || arg === '-h') {
      parsed.help = true;
      continue;
    }

    if (arg === '--input') {
      parsed.inputs.push(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg.startsWith('--input=')) {
      parsed.inputs.push(arg.substring('--input='.length));
      continue;
    }

    if (arg === '--output') {
      parsed.output = argv[i + 1];
      i += 1;
      continue;
    }
    if (arg.startsWith('--output=')) {
      parsed.output = arg.substring('--output='.length);
      continue;
    }

    if (arg === '--section') {
      parsed.section = argv[i + 1] || DEFAULT_SECTION;
      i += 1;
      continue;
    }
    if (arg.startsWith('--section=')) {
      parsed.section = arg.substring('--section='.length);
      continue;
    }

    if (arg === '--chart') {
      parsed.chart = argv[i + 1];
      i += 1;
      continue;
    }
    if (arg.startsWith('--chart=')) {
      parsed.chart = arg.substring('--chart='.length);
      continue;
    }

    if (arg === '--metrics') {
      parsed.metricIds = String(argv[i + 1] || '')
        .split(',')
        .map((item) => item.trim())
        .filter((item) => item.length > 0);
      i += 1;
      continue;
    }
    if (arg.startsWith('--metrics=')) {
      parsed.metricIds = arg.substring('--metrics='.length)
        .split(',')
        .map((item) => item.trim())
        .filter((item) => item.length > 0);
      continue;
    }

    if (arg === '--width') {
      parsed.width = Number.parseInt(argv[i + 1], 10);
      i += 1;
      continue;
    }
    if (arg.startsWith('--width=')) {
      parsed.width = Number.parseInt(arg.substring('--width='.length), 10);
      continue;
    }

    if (arg === '--height') {
      parsed.height = Number.parseInt(argv[i + 1], 10);
      i += 1;
      continue;
    }
    if (arg.startsWith('--height=')) {
      parsed.height = Number.parseInt(arg.substring('--height='.length), 10);
      continue;
    }

    if (arg.startsWith('--')) {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (parsed.inputs.length === 0) {
    throw new Error('Missing required --input path.');
  }
  if (!Number.isFinite(parsed.width) || parsed.width < 300) {
    throw new Error('--width must be a number >= 300.');
  }
  if (!Number.isFinite(parsed.height) || parsed.height < 260) {
    throw new Error('--height must be a number >= 260.');
  }
  if (!CHART_TYPES.includes(parsed.chart)) {
    throw new Error(`--chart must be one of ${CHART_TYPES.join(', ')}`);
  }

  return parsed;
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function toNumber(value) {
  if (isFiniteNumber(value)) return value;
  if (!value || typeof value !== 'object') return null;
  if (isFiniteNumber(value.mean)) return value.mean;
  if (isFiniteNumber(value.median)) return value.median;
  if (isFiniteNumber(value.min)) return value.min;
  if (isFiniteNumber(value.max)) return value.max;
  return null;
}

function getPathValue(obj, pathExpr) {
  let cursor = obj;
  for (const segment of pathExpr.split('.').filter((segment) => segment.length > 0)) {
    if (!cursor || typeof cursor !== 'object') return null;
    cursor = cursor[segment];
  }
  return cursor;
}

function firstFinite(values) {
  for (const value of values) {
    const num = toNumber(value);
    if (isFiniteNumber(num)) return num;
  }
  return null;
}

function getFirstFinitePathValue(obj, pathExpressions) {
  return firstFinite((pathExpressions || []).map((expr) => getPathValue(obj, expr)));
}

function sectionHasComparable(section) {
  if (!section || typeof section !== 'object') return false;
  return Boolean(section?.doppler || section?.tjs || section?.transformersjs || section?.result);
}

function resolveSection(report, requestedSection) {
  if (!report || typeof report !== 'object') return null;
  if (!report.sections) {
    return sectionHasComparable(report)
      ? { section: requestedSection, payload: report }
      : null;
  }

  const candidates = [
    requestedSection,
    ...FALLBACK_SECTIONS,
  ];
  const seen = new Set();

  for (const candidate of candidates) {
    if (!candidate) continue;
    if (seen.has(candidate)) continue;
    seen.add(candidate);
    let cursor = report.sections;
    const segments = candidate.split('/').map((segment) => segment.trim()).filter((segment) => segment.length > 0);
    for (const segment of segments) {
      if (!cursor || typeof cursor !== 'object') {
        cursor = null;
        break;
      }
      cursor = cursor[segment];
    }
    if (sectionHasComparable(cursor)) {
      return { section: candidate, payload: cursor };
    }
  }

  return null;
}

function enginePayload(sectionPayload, engineId) {
  if (!sectionPayload || typeof sectionPayload !== 'object') return null;
  if (engineId === 'tjs') {
    return sectionPayload.tjs ?? sectionPayload.transformersjs ?? null;
  }
  return sectionPayload.doppler ?? sectionPayload[engineId] ?? sectionPayload.result ?? null;
}

function resolveMetric(payload, metricDef, engineId) {
  if (!payload || typeof payload !== 'object') {
    return { value: null, status: 'missing' };
  }
  if (payload.failed === true) {
    return {
      value: null,
      status: 'failed',
      error: payload.error?.message || payload.error || 'failed',
    };
  }

  const scope = payload.result ? payload : { result: payload };
  const derivedKey = DERIVED_KEY[engineId] || engineId;
  const derivedSpec = metricDef.derived?.[derivedKey];

  if (derivedSpec?.numeratorPaths || derivedSpec?.denominatorPaths) {
    const numerator = getFirstFinitePathValue(scope, derivedSpec.numeratorPaths || []);
    const denominator = getFirstFinitePathValue(scope, derivedSpec.denominatorPaths || []);
    if (isFiniteNumber(numerator) && isFiniteNumber(denominator) && denominator !== 0) {
      return { value: numerator / denominator, status: 'ok' };
    }
  }

  const hintPaths = METRIC_PATH_HINTS[metricDef.id] || [];
  const value = getFirstFinitePathValue(scope, hintPaths);
  if (value !== null) return { value, status: 'ok' };

  const direct = scope.result?.[metricDef.id] ?? scope[metricDef.id];
  const directValue = toNumber(direct);
  return { value: directValue, status: directValue === null ? 'missing' : 'ok' };
}

function metricRowsFromReport(report, sectionPayload, metricIds) {
  const sourceMetrics = Array.isArray(report.metricContract?.metrics) && report.metricContract.metrics.length > 0
    ? report.metricContract.metrics
    : DEFAULT_METRICS;
  const requested = metricIds && metricIds.length > 0
    ? new Set(metricIds)
    : null;

  const selected = requested
    ? sourceMetrics.filter((metric) => requested.has(metric.id))
    : sourceMetrics;

  return selected.length > 0 ? selected : sourceMetrics;
}

function collectRows(report, sectionPayload, metricIds) {
  const metrics = metricRowsFromReport(report, sectionPayload, metricIds);
  const dopplerPayload = enginePayload(sectionPayload, 'doppler');
  const tjsPayload = enginePayload(sectionPayload, 'tjs');

  return metrics.map((metricDef) => {
    const doppler = resolveMetric(dopplerPayload, metricDef, 'doppler');
    const tjs = resolveMetric(tjsPayload, metricDef, 'tjs');
    return {
      id: metricDef.id,
      label: metricDef.label || metricDef.id,
      unit: metricDef.unit || '',
      higherBetter: metricDef.higherBetter !== false,
      doppler,
      tjs,
    };
  });
}

function hasAnyValue(rows) {
  return rows.some((row) => isFiniteNumber(row.doppler.value) || isFiniteNumber(row.tjs.value));
}

function formatValue(value, unit) {
  if (!isFiniteNumber(value)) return 'n/a';
  if (Math.abs(value) >= 1000) return `${value.toFixed(1)} ${unit}`.trim();
  return `${value.toFixed(2)} ${unit}`.trim();
}

function escapeXml(raw) {
  return String(raw)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function resolveScale(rows) {
  const scales = new Map();
  for (const row of rows) {
    const values = [row.doppler.value, row.tjs.value].filter((value) => isFiniteNumber(value));
    if (values.length === 0) {
      scales.set(row.id, { min: null, max: null, hasValue: false });
      continue;
    }
    const min = Math.min(...values);
    const max = Math.max(...values);
    scales.set(row.id, { min, max, hasValue: true });
  }
  return scales;
}

function scoreMetric(row, scale, engineId) {
  const value = row[engineId].value;
  if (!scale || !scale.hasValue || !isFiniteNumber(value)) return 0;
  if (row.higherBetter) {
    if (scale.max <= 0) return 0;
    return Math.max(0, Math.min(1, value / scale.max));
  }
  if (value <= 0) return 1;
  if (scale.min <= 0) return 0;
  return Math.max(0, Math.min(1, scale.min / value));
}

function buildScaledRows(rows) {
  const scales = resolveScale(rows);
  const enriched = rows.map((row) => {
    const dopplerScore = scoreMetric(row, scales.get(row.id), 'doppler');
    const tjsScore = scoreMetric(row, scales.get(row.id), 'tjs');
    return { ...row, dopplerScore, tjsScore };
  });
  return enriched;
}

function renderBarChart(rows, width, height, title, subtitle, sectionLabel) {
  const left = 240;
  const right = 52;
  const top = 76;
  const rowGap = (height - 170) / Math.max(rows.length, 1);
  const barAreaMax = width - left - right;
  const baseY = 84;
  let body = '';

  rows.forEach((metric, index) => {
    const y = baseY + index * rowGap;
    const maxValue = Math.max(
      1,
      metric.doppler.value ?? 1,
      metric.tjs.value ?? 1,
      ...(isFiniteNumber(metric.doppler.value) ? [metric.doppler.value] : []),
      ...(isFiniteNumber(metric.tjs.value) ? [metric.tjs.value] : []),
    );
    const dopplerWidth = isFiniteNumber(metric.doppler.value) ? (metric.doppler.value / maxValue) * barAreaMax : 0;
    const tjsWidth = isFiniteNumber(metric.tjs.value) ? (metric.tjs.value / maxValue) * barAreaMax : 0;
    const rowCenter = y + 34;

    body += `<text x="32" y="${y + 6}" fill="${PALETTE.text}" font-family="monospace" font-size="13">${escapeXml(metric.label)}</text>\n`;
    body += `<text x="32" y="${y + 22}" fill="${PALETTE.muted}" font-family="monospace" font-size="11">${metric.higherBetter ? 'higher is better' : 'lower is better'}</text>\n`;
    body += `<line x1="${left}" y1="${y + 44}" x2="${left + barAreaMax}" y2="${y + 44}" stroke="${PALETTE.grid}" stroke-width="1" />\n`;

    if (metric.doppler.status === 'ok') {
      body += `<rect x="${left + 80}" y="${rowCenter - 18}" width="${dopplerWidth}" height="16" fill="${PALETTE.doppler}" />\n`;
      body += `<text x="${left + 88 + dopplerWidth}" y="${rowCenter - 5}" fill="${PALETTE.doppler}" font-family="monospace" font-size="11">${formatValue(metric.doppler.value, metric.unit)}</text>\n`;
      if (!isFiniteNumber(metric.doppler.value)) {
        body += `<text x="${left + 90}" y="${rowCenter - 5}" fill="${PALETTE.text}" font-family="monospace" font-size="11">n/a</text>\n`;
      }
    } else {
      body += `<rect x="${left + 80}" y="${rowCenter - 18}" width="${barAreaMax}" height="16" fill="${PALETTE.failFill}" />\n`;
      body += `<text x="${left + 90}" y="${rowCenter - 5}" fill="${PALETTE.text}" font-family="monospace" font-size="11">Doppler.js failed</text>\n`;
    }

    if (metric.tjs.status === 'ok') {
      body += `<rect x="${left + 80}" y="${rowCenter + 2}" width="${tjsWidth}" height="16" fill="${PALETTE.tjs}" />\n`;
      body += `<text x="${left + 88 + tjsWidth}" y="${rowCenter + 15}" fill="${PALETTE.tjs}" font-family="monospace" font-size="11">${formatValue(metric.tjs.value, metric.unit)}</text>\n`;
      if (!isFiniteNumber(metric.tjs.value)) {
        body += `<text x="${left + 90}" y="${rowCenter + 15}" fill="${PALETTE.text}" font-family="monospace" font-size="11">n/a</text>\n`;
      }
    } else {
      body += `<rect x="${left + 80}" y="${rowCenter + 2}" width="${barAreaMax}" height="16" fill="${PALETTE.failFill}" />\n`;
      body += `<text x="${left + 90}" y="${rowCenter + 15}" fill="${PALETTE.text}" font-family="monospace" font-size="11">Transformers.js failed</text>\n`;
    }
  });

  body += `<text x="${left + 80}" y="${top - 24}" fill="${PALETTE.doppler}" font-family="monospace" font-size="11">${ENGINE_META.doppler.label}</text>\n`;
  body += `<text x="${left + 170}" y="${top - 24}" fill="${PALETTE.tjs}" font-family="monospace" font-size="11">${ENGINE_META.tjs.label}</text>\n`;
  body += `<text x="40" y="36" fill="${PALETTE.text}" font-family="monospace" font-size="18" font-weight="bold">${escapeXml(title)}</text>\n`;
  body += `<text x="40" y="54" fill="${PALETTE.muted}" font-family="monospace" font-size="12">${escapeXml(subtitle)}</text>\n`;
  body += `<text x="40" y="${height - 22}" fill="${PALETTE.muted}" font-family="monospace" font-size="11">${escapeXml(`Section: ${sectionLabel}`)}</text>\n`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="${width}" height="${height}" fill="${PALETTE.bg}" />
  ${body}
</svg>`;
}

function renderStackedBars(rows, width, height, title, subtitle, sectionLabel) {
  const chartLeft = 220;
  const chartRight = 44;
  const barMaxWidth = width - chartLeft - chartRight;
  const baseY = 102;
  const rowHeight = 110;
  const top = 84;
  const engines = [ENGINE_META.doppler, ENGINE_META.tjs];
  const scaledRows = buildScaledRows(rows);
  let body = '';
  const scale = resolveScale(rows);

  engines.forEach((engine, index) => {
    const y = top + index * rowHeight;
    const label = `${engine.label} (normalized metric composition)`;
    const scoreTotal = scaledRows.reduce((sum, metric) => sum + (scoreMetric(metric, scale.get(metric.id), engine.key)), 0);
    const base = baseY + index * rowHeight;

    body += `<text x="32" y="${y + 8}" fill="${PALETTE.text}" font-family="monospace" font-size="14" font-weight="bold">${escapeXml(label)}</text>\n`;
    body += `<line x1="${chartLeft}" y1="${base + 26}" x2="${chartLeft + barMaxWidth}" y2="${base + 26}" stroke="${PALETTE.grid}" stroke-width="1" />\n`;

    if (scoreTotal <= 0) {
      body += `<text x="${chartLeft + 90}" y="${base + 18}" fill="${PALETTE.muted}" font-family="monospace" font-size="11">${engine.label} has no valid metric values.</text>\n`;
      return;
    }

    let cursor = chartLeft + 80;
    scaledRows.forEach((metric, i) => {
      const metricScore = scoreMetric(metric, scale.get(metric.id), engine.key);
      const segmentWidth = (metricScore / scoreTotal) * barMaxWidth;
      const fill = PALETTE.metric[i % PALETTE.metric.length];
      body += `<rect x="${cursor}" y="${base + 8}" width="${segmentWidth}" height="24" fill="${fill}" />\n`;
      const label = `${metric.label}: ${formatValue(metric[engine.key].value, metric.unit)}`;
      const safeLabel = escapeXml(label.length > 34 ? `${label.substring(0, 31)}...` : label);
      if (segmentWidth > 40) {
        body += `<text x="${cursor + 4}" y="${base + 23}" fill="#000" font-family="monospace" font-size="10">${safeLabel}</text>\n`;
      }
      cursor += segmentWidth;
    });
  });

  rows.forEach((metric, index) => {
    const x = 32 + (index % 2) * ((width - 64) / 2);
    const y = height - 50 + Math.floor(index / 2) * 16;
    const fill = PALETTE.metric[index % PALETTE.metric.length];
    body += `<rect x="${x}" y="${y}" width="12" height="12" fill="${fill}" />\n`;
    body += `<text x="${x + 16}" y="${y + 10}" fill="${PALETTE.text}" font-family="monospace" font-size="10">${escapeXml(metric.label)}</text>\n`;
  });

  body += `<text x="40" y="36" fill="${PALETTE.text}" font-family="monospace" font-size="18" font-weight="bold">${escapeXml(title)}</text>\n`;
  body += `<text x="40" y="54" fill="${PALETTE.muted}" font-family="monospace" font-size="12">${escapeXml(subtitle)}</text>\n`;
  body += `<text x="40" y="${height - 22}" fill="${PALETTE.muted}" font-family="monospace" font-size="11">${escapeXml(`Section: ${sectionLabel}`)}</text>\n`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="${width}" height="${height}" fill="${PALETTE.bg}" />
  ${body}
</svg>`;
}

const RADAR_DEFAULT_METRIC_IDS = Object.freeze(new Set([
  'firstTokenMs',
  'totalRunMs',
  'firstResponseMs',
  'decodeMs',
  'modelLoadMs',
]));

function filterRadarRows(rows, metricIds) {
  if (metricIds && metricIds.length > 0) return rows;
  return rows.filter((row) => RADAR_DEFAULT_METRIC_IDS.has(row.id));
}

function renderRadar(rows, width, height, title, subtitle, sectionLabel) {
  const centerX = width / 2;
  const centerY = height / 2 + 16;
  const radius = Math.min(width, height - 170) / 2 - 110;
  const ringCount = 4;
  const axisCount = Math.max(rows.length, 1);
  const angleStep = (Math.PI * 2) / axisCount;
  const scale = resolveScale(rows);
  const scaledRows = buildScaledRows(rows);
  let body = '';

  const rings = [];
  for (let ring = 1; ring <= ringCount; ring += 1) {
    const r = (radius / ringCount) * ring;
    const pts = rows.map((metric, i) => {
      const angle = -Math.PI / 2 + i * angleStep;
      return `${centerX + r * Math.cos(angle)},${centerY + r * Math.sin(angle)}`;
    }).join(' ');
    rings.push(pts);
  }
  for (const pts of rings) {
    body += `<polygon points="${pts}" fill="none" stroke="${PALETTE.grid}" stroke-width="1" />\n`;
  }

  rows.forEach((metric, i) => {
    const angle = -Math.PI / 2 + i * angleStep;
    const labelRadius = radius + 34;
    const x = centerX + labelRadius * Math.cos(angle);
    const y = centerY + labelRadius * Math.sin(angle);
    body += `<line x1="${centerX}" y1="${centerY}" x2="${centerX + radius * Math.cos(angle)}" y2="${centerY + radius * Math.sin(angle)}" stroke="${PALETTE.grid}" stroke-width="1" />\n`;
    const anchor = Math.abs(x - centerX) < 10 ? 'middle' : x < centerX ? 'end' : 'start';
    body += `<text x="${x}" y="${y}" fill="${PALETTE.text}" text-anchor="${anchor}" font-family="monospace" font-size="11">${escapeXml(metric.label)}</text>\n`;
  });

  const engines = [ENGINE_META.doppler, ENGINE_META.tjs];
  engines.forEach((engine, engineIndex) => {
    const points = rows.map((metric, i) => {
      const angle = -Math.PI / 2 + i * angleStep;
      const score = engine.key === 'doppler'
        ? scaledRows[i].dopplerScore
        : scaledRows[i].tjsScore;
      const r = radius * score;
      return `${centerX + r * Math.cos(angle)},${centerY + r * Math.sin(angle)}`;
    }).join(' ');
    const yLegend = 56 + engineIndex * 16;
    const fillOpacity = engine.key === 'doppler' ? '0.4' : '0.25';
    body += `<polygon points="${points}" fill="${engine.color}" fill-opacity="${fillOpacity}" stroke="${engine.color}" stroke-width="2" />\n`;
    body += `<rect x="${width - 190}" y="${yLegend - 8}" width="12" height="12" fill="${engine.color}" />\n`;
    body += `<text x="${width - 170}" y="${yLegend + 1}" fill="${PALETTE.text}" font-family="monospace" font-size="12">${engine.label}</text>\n`;
  });

  const allBad = engines.every((engine) => {
    const metricScores = rows.map((metric) => {
      return engine.key === 'doppler'
        ? scoreMetric(metric, scale.get(metric.id), 'doppler')
        : scoreMetric(metric, scale.get(metric.id), 'tjs');
    });
    return metricScores.every((value) => value <= 0);
  });
  if (allBad) {
    body += `<text x="${centerX - 120}" y="${centerY}" fill="${PALETTE.muted}" font-family="monospace" font-size="12">No comparable valid values for this section.</text>\n`;
  }

  body += `<text x="40" y="36" fill="${PALETTE.text}" font-family="monospace" font-size="18" font-weight="bold">${escapeXml(title)}</text>\n`;
  body += `<text x="40" y="54" fill="${PALETTE.muted}" font-family="monospace" font-size="12">${escapeXml(subtitle)}</text>\n`;
  body += `<text x="40" y="70" fill="${PALETTE.muted}" font-family="monospace" font-size="11">All axes are latency (ms), inverted so bigger polygon = faster.</text>\n`;
  body += `<text x="40" y="${height - 22}" fill="${PALETTE.muted}" font-family="monospace" font-size="11">${escapeXml(`Section: ${sectionLabel}`)}</text>\n`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="${width}" height="${height}" fill="${PALETTE.bg}" />
  ${body}
</svg>`;
}

const PHASE_COLORS = Object.freeze({
  warmLoad: '#ef4444',
  prefill: '#fbbf24',
  ttftMarker: '#ffffff',
  decode: '#3b82f6',
});

function resolvePhaseValues(sectionPayload, engineId) {
  const payload = enginePayload(sectionPayload, engineId);
  if (!payload || payload.failed === true) return null;
  const scope = payload.result ? payload : { result: payload };
  const timing = scope.result?.timing || scope.result || {};

  const modelLoadMs = toNumber(timing.modelLoadMs);
  const prefillMs = toNumber(timing.prefillMs);
  const firstTokenMs = toNumber(timing.firstTokenMs);
  const decodeMs = toNumber(timing.decodeMs);

  const ttft = isFiniteNumber(firstTokenMs) ? firstTokenMs : prefillMs;

  const endToEnd = [modelLoadMs, ttft, decodeMs]
    .filter((v) => isFiniteNumber(v))
    .reduce((sum, v) => sum + v, 0);

  return { modelLoadMs, ttft, prefillMs, decodeMs, endToEnd };
}

function renderPhases(rows, width, height, title, subtitle, sectionLabel, sectionPayload) {
  const engines = [ENGINE_META.doppler, ENGINE_META.tjs];
  const phaseData = new Map();
  let globalMax = 0;

  for (const engine of engines) {
    const resolved = resolvePhaseValues(sectionPayload, engine.key);
    phaseData.set(engine.key, resolved);
    if (resolved && resolved.endToEnd > globalMax) {
      globalMax = resolved.endToEnd;
    }
  }

  if (globalMax <= 0) globalMax = 1;

  const left = 200;
  const right = 120;
  const barAreaMax = width - left - right;
  const barHeight = 36;
  const engineGap = 100;
  const baseY = 100;
  let body = '';

  engines.forEach((engine, engineIndex) => {
    const y = baseY + engineIndex * engineGap;
    const resolved = phaseData.get(engine.key);

    body += `<text x="32" y="${y + barHeight / 2 + 5}" fill="${engine.color}" font-family="monospace" font-size="14" font-weight="bold">${engine.label}</text>\n`;

    if (!resolved) {
      body += `<rect x="${left}" y="${y}" width="${barAreaMax}" height="${barHeight}" fill="${PALETTE.failFill}" />\n`;
      body += `<text x="${left + 12}" y="${y + barHeight / 2 + 4}" fill="${PALETTE.text}" font-family="monospace" font-size="12">No data</text>\n`;
      return;
    }

    const pxPerMs = barAreaMax / globalMax;
    let cursor = left;

    if (isFiniteNumber(resolved.modelLoadMs) && resolved.modelLoadMs > 0) {
      const w = resolved.modelLoadMs * pxPerMs;
      body += `<rect x="${cursor}" y="${y}" width="${w}" height="${barHeight}" fill="${PHASE_COLORS.warmLoad}" />\n`;
      if (w > 80) {
        body += `<text x="${cursor + 6}" y="${y + barHeight / 2 + 4}" fill="#000" font-family="monospace" font-size="11" font-weight="bold">OPFS to VRAM</text>\n`;
        body += `<text x="${cursor + 6}" y="${y + barHeight / 2 + 16}" fill="#000" font-family="monospace" font-size="10">${resolved.modelLoadMs.toFixed(1)} ms</text>\n`;
      }
      cursor += w;
    }

    if (isFiniteNumber(resolved.ttft) && resolved.ttft > 0) {
      const ttftW = resolved.ttft * pxPerMs;
      const ttftX = cursor;
      body += `<rect x="${ttftX}" y="${y}" width="${ttftW}" height="${barHeight}" fill="${PHASE_COLORS.prefill}" />\n`;

      if (ttftW > 80) {
        body += `<text x="${ttftX + 6}" y="${y + barHeight / 2 + 4}" fill="#000" font-family="monospace" font-size="11" font-weight="bold">Prefill ${resolved.ttft.toFixed(1)} ms</text>\n`;
      } else if (ttftW > 40) {
        body += `<text x="${ttftX + 4}" y="${y + barHeight / 2 + 4}" fill="#000" font-family="monospace" font-size="10">${resolved.ttft.toFixed(0)}</text>\n`;
      }

      const ttftMarkerX = ttftX + ttftW;
      const markerSize = 5;
      body += `<polygon points="${ttftMarkerX},${y - 2} ${ttftMarkerX - markerSize},${y - markerSize - 4} ${ttftMarkerX + markerSize},${y - markerSize - 4}" fill="${PHASE_COLORS.ttftMarker}" />\n`;
      body += `<text x="${ttftMarkerX}" y="${y - markerSize - 6}" fill="${PALETTE.muted}" font-family="monospace" font-size="9" text-anchor="middle">TTFT</text>\n`;

      cursor += ttftW;
    }

    if (isFiniteNumber(resolved.decodeMs) && resolved.decodeMs > 0) {
      const w = resolved.decodeMs * pxPerMs;
      body += `<rect x="${cursor}" y="${y}" width="${w}" height="${barHeight}" fill="${PHASE_COLORS.decode}" />\n`;
      if (w > 50) {
        body += `<text x="${cursor + 6}" y="${y + barHeight / 2 + 4}" fill="#000" font-family="monospace" font-size="11" font-weight="bold">Decode</text>\n`;
        body += `<text x="${cursor + 6}" y="${y + barHeight / 2 + 16}" fill="#000" font-family="monospace" font-size="10">${resolved.decodeMs.toFixed(1)} ms</text>\n`;
      }
      cursor += w;
    }

    body += `<text x="${cursor + 8}" y="${y + barHeight / 2 + 5}" fill="${PALETTE.text}" font-family="monospace" font-size="12">${resolved.endToEnd.toFixed(1)} ms</text>\n`;
  });

  const legendY = baseY + engines.length * engineGap + 24;
  const legendItems = [
    { id: 'warmLoad', label: 'OPFS to VRAM (Warm Load)' },
    { id: 'prefill', label: 'Prefill' },
    { id: 'decode', label: 'Decode' },
  ];
  legendItems.forEach((item, i) => {
    const x = left + i * 210;
    body += `<rect x="${x}" y="${legendY}" width="14" height="14" fill="${PHASE_COLORS[item.id]}" />\n`;
    body += `<text x="${x + 20}" y="${legendY + 11}" fill="${PALETTE.text}" font-family="monospace" font-size="11">${item.label}</text>\n`;
  });
  const ttftLegendX = left + legendItems.length * 210;
  body += `<polygon points="${ttftLegendX + 7},${legendY} ${ttftLegendX + 2},${legendY + 7} ${ttftLegendX + 12},${legendY + 7}" fill="${PHASE_COLORS.ttftMarker}" />\n`;
  body += `<text x="${ttftLegendX + 20}" y="${legendY + 11}" fill="${PALETTE.text}" font-family="monospace" font-size="11">TTFT marker</text>\n`;

  body += `<text x="40" y="36" fill="${PALETTE.text}" font-family="monospace" font-size="18" font-weight="bold">${escapeXml(title)}</text>\n`;
  body += `<text x="40" y="54" fill="${PALETTE.muted}" font-family="monospace" font-size="12">${escapeXml(subtitle)}</text>\n`;
  body += `<text x="40" y="${height - 22}" fill="${PALETTE.muted}" font-family="monospace" font-size="11">${escapeXml(`Section: ${sectionLabel}`)}</text>\n`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="${width}" height="${height}" fill="${PALETTE.bg}" />
  ${body}
</svg>`;
}

function renderMultiPhases(entries, width, title, subtitle) {
  const engines = [ENGINE_META.doppler, ENGINE_META.tjs];
  const left = 200;
  const right = 120;
  const barAreaMax = width - left - right;
  const barHeight = 36;
  const engineGap = 56;
  const workloadGap = 30;
  const baseY = 100;
  const workloadBlockHeight = engines.length * engineGap + workloadGap;

  let globalMax = 0;
  const workloads = entries.map((entry) => {
    const phaseData = new Map();
    for (const engine of engines) {
      const resolved = resolvePhaseValues(entry.sectionPayload, engine.key);
      phaseData.set(engine.key, resolved);
      if (resolved && resolved.endToEnd > globalMax) {
        globalMax = resolved.endToEnd;
      }
    }
    const workloadLabel = prettifyWorkload(entry.report.workload) || entry.report.workload?.id || path.basename(entry.inputPath, '.json');
    return { phaseData, workloadLabel, sectionLabel: entry.resolvedSection };
  });

  if (globalMax <= 0) globalMax = 1;

  const legendHeight = 50;
  const height = baseY + workloads.length * workloadBlockHeight + legendHeight + 40;
  let body = '';

  workloads.forEach((workload, wIndex) => {
    const workloadY = baseY + wIndex * workloadBlockHeight;

    body += `<text x="32" y="${workloadY - 8}" fill="${PALETTE.muted}" font-family="monospace" font-size="12" font-weight="bold">${escapeXml(workload.workloadLabel)}</text>\n`;

    engines.forEach((engine, engineIndex) => {
      const y = workloadY + engineIndex * engineGap;
      const resolved = workload.phaseData.get(engine.key);

      body += `<text x="32" y="${y + barHeight / 2 + 5}" fill="${engine.color}" font-family="monospace" font-size="14" font-weight="bold">${engine.label}</text>\n`;

      if (!resolved) {
        body += `<rect x="${left}" y="${y}" width="${barAreaMax}" height="${barHeight}" fill="${PALETTE.failFill}" />\n`;
        body += `<text x="${left + 12}" y="${y + barHeight / 2 + 4}" fill="${PALETTE.text}" font-family="monospace" font-size="12">No data</text>\n`;
        return;
      }

      const pxPerMs = barAreaMax / globalMax;
      let cursor = left;

      if (isFiniteNumber(resolved.modelLoadMs) && resolved.modelLoadMs > 0) {
        const w = resolved.modelLoadMs * pxPerMs;
        body += `<rect x="${cursor}" y="${y}" width="${w}" height="${barHeight}" fill="${PHASE_COLORS.warmLoad}" />\n`;
        if (w > 80) {
          body += `<text x="${cursor + 6}" y="${y + barHeight / 2 + 4}" fill="#000" font-family="monospace" font-size="11" font-weight="bold">OPFS to VRAM</text>\n`;
          body += `<text x="${cursor + 6}" y="${y + barHeight / 2 + 16}" fill="#000" font-family="monospace" font-size="10">${resolved.modelLoadMs.toFixed(1)} ms</text>\n`;
        }
        cursor += w;
      }

      if (isFiniteNumber(resolved.ttft) && resolved.ttft > 0) {
        const ttftW = resolved.ttft * pxPerMs;
        const ttftX = cursor;
        body += `<rect x="${ttftX}" y="${y}" width="${ttftW}" height="${barHeight}" fill="${PHASE_COLORS.prefill}" />\n`;
        if (ttftW > 80) {
          body += `<text x="${ttftX + 6}" y="${y + barHeight / 2 + 4}" fill="#000" font-family="monospace" font-size="11" font-weight="bold">Prefill ${resolved.ttft.toFixed(1)} ms</text>\n`;
        } else if (ttftW > 40) {
          body += `<text x="${ttftX + 4}" y="${y + barHeight / 2 + 4}" fill="#000" font-family="monospace" font-size="10">${resolved.ttft.toFixed(0)}</text>\n`;
        }
        const ttftMarkerX = ttftX + ttftW;
        const markerSize = 5;
        body += `<polygon points="${ttftMarkerX},${y - 2} ${ttftMarkerX - markerSize},${y - markerSize - 4} ${ttftMarkerX + markerSize},${y - markerSize - 4}" fill="${PHASE_COLORS.ttftMarker}" />\n`;
        body += `<text x="${ttftMarkerX}" y="${y - markerSize - 6}" fill="${PALETTE.muted}" font-family="monospace" font-size="9" text-anchor="middle">TTFT</text>\n`;
        cursor += ttftW;
      }

      if (isFiniteNumber(resolved.decodeMs) && resolved.decodeMs > 0) {
        const w = resolved.decodeMs * pxPerMs;
        body += `<rect x="${cursor}" y="${y}" width="${w}" height="${barHeight}" fill="${PHASE_COLORS.decode}" />\n`;
        if (w > 50) {
          body += `<text x="${cursor + 6}" y="${y + barHeight / 2 + 4}" fill="#000" font-family="monospace" font-size="11" font-weight="bold">Decode</text>\n`;
          body += `<text x="${cursor + 6}" y="${y + barHeight / 2 + 16}" fill="#000" font-family="monospace" font-size="10">${resolved.decodeMs.toFixed(1)} ms</text>\n`;
        }
        cursor += w;
      }

      body += `<text x="${cursor + 8}" y="${y + barHeight / 2 + 5}" fill="${PALETTE.text}" font-family="monospace" font-size="12">${resolved.endToEnd.toFixed(1)} ms</text>\n`;
    });
  });

  const legendY = baseY + workloads.length * workloadBlockHeight;
  const legendItems = [
    { id: 'warmLoad', label: 'OPFS to VRAM (Warm Load)' },
    { id: 'prefill', label: 'Prefill' },
    { id: 'decode', label: 'Decode' },
  ];
  legendItems.forEach((item, i) => {
    const x = left + i * 210;
    body += `<rect x="${x}" y="${legendY}" width="14" height="14" fill="${PHASE_COLORS[item.id]}" />\n`;
    body += `<text x="${x + 20}" y="${legendY + 11}" fill="${PALETTE.text}" font-family="monospace" font-size="11">${item.label}</text>\n`;
  });
  const ttftLegendX = left + legendItems.length * 210;
  body += `<polygon points="${ttftLegendX + 7},${legendY} ${ttftLegendX + 2},${legendY + 7} ${ttftLegendX + 12},${legendY + 7}" fill="${PHASE_COLORS.ttftMarker}" />\n`;
  body += `<text x="${ttftLegendX + 20}" y="${legendY + 11}" fill="${PALETTE.text}" font-family="monospace" font-size="11">TTFT marker</text>\n`;

  body += `<text x="40" y="36" fill="${PALETTE.text}" font-family="monospace" font-size="18" font-weight="bold">${escapeXml(title)}</text>\n`;
  body += `<text x="40" y="54" fill="${PALETTE.muted}" font-family="monospace" font-size="12">${escapeXml(subtitle)}</text>\n`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="${width}" height="${height}" fill="${PALETTE.bg}" />
  ${body}
</svg>`;
}

function renderMultiRadar(entries, perRadarHeight, title, subtitle, metricIds) {
  const numWorkloads = entries.length;
  const perRadarWidth = DEFAULT_WIDTH;
  const totalWidth = perRadarWidth * numWorkloads;

  const allScaledData = entries.map((entry) => {
    const allRows = collectRows(entry.report, entry.sectionPayload, metricIds);
    const rows = filterRadarRows(allRows, metricIds);
    const scaledRows = buildScaledRows(rows);
    const workloadLabel = prettifyWorkload(entry.report.workload) || entry.report.workload?.id || path.basename(entry.inputPath, '.json');
    return { rows, scaledRows, workloadLabel };
  });

  let body = '';
  const headerHeight = 90;
  const legendHeight = 40;
  const height = perRadarHeight + headerHeight + legendHeight;

  allScaledData.forEach((data, wIndex) => {
    const offsetX = wIndex * perRadarWidth;
    const centerX = offsetX + perRadarWidth / 2;
    const centerY = headerHeight + (perRadarHeight - 40) / 2;
    const radius = Math.min(perRadarWidth, perRadarHeight - 40) / 2 - 110;
    const ringCount = 4;
    const axisCount = Math.max(data.rows.length, 1);
    const angleStep = (Math.PI * 2) / axisCount;
    const scale = resolveScale(data.rows);

    for (let ring = 1; ring <= ringCount; ring += 1) {
      const r = (radius / ringCount) * ring;
      const pts = data.rows.map((_, i) => {
        const angle = -Math.PI / 2 + i * angleStep;
        return `${centerX + r * Math.cos(angle)},${centerY + r * Math.sin(angle)}`;
      }).join(' ');
      body += `<polygon points="${pts}" fill="none" stroke="${PALETTE.grid}" stroke-width="1" />\n`;
    }

    data.rows.forEach((metric, i) => {
      const angle = -Math.PI / 2 + i * angleStep;
      const labelRadius = radius + 34;
      const x = centerX + labelRadius * Math.cos(angle);
      const y = centerY + labelRadius * Math.sin(angle);
      body += `<line x1="${centerX}" y1="${centerY}" x2="${centerX + radius * Math.cos(angle)}" y2="${centerY + radius * Math.sin(angle)}" stroke="${PALETTE.grid}" stroke-width="1" />\n`;
      const anchor = Math.abs(x - centerX) < 10 ? 'middle' : x < centerX ? 'end' : 'start';
      body += `<text x="${x}" y="${y}" fill="${PALETTE.text}" text-anchor="${anchor}" font-family="monospace" font-size="11">${escapeXml(metric.label)}</text>\n`;
    });

    const engines = [ENGINE_META.doppler, ENGINE_META.tjs];
    engines.forEach((engine) => {
      const points = data.rows.map((_, i) => {
        const angle = -Math.PI / 2 + i * angleStep;
        const score = engine.key === 'doppler'
          ? data.scaledRows[i].dopplerScore
          : data.scaledRows[i].tjsScore;
        const r = radius * score;
        return `${centerX + r * Math.cos(angle)},${centerY + r * Math.sin(angle)}`;
      }).join(' ');
      const fillOpacity = engine.key === 'doppler' ? '0.4' : '0.25';
      body += `<polygon points="${points}" fill="${engine.color}" fill-opacity="${fillOpacity}" stroke="${engine.color}" stroke-width="2" />\n`;
    });

    body += `<text x="${centerX}" y="${height - legendHeight - 4}" fill="${PALETTE.muted}" font-family="monospace" font-size="12" text-anchor="middle">${escapeXml(data.workloadLabel)}</text>\n`;
  });

  const engines = [ENGINE_META.doppler, ENGINE_META.tjs];
  engines.forEach((engine, i) => {
    const legendX = totalWidth - 200;
    const legendY = headerHeight - 30 + i * 16;
    body += `<rect x="${legendX}" y="${legendY - 8}" width="12" height="12" fill="${engine.color}" />\n`;
    body += `<text x="${legendX + 18}" y="${legendY + 1}" fill="${PALETTE.text}" font-family="monospace" font-size="12">${engine.label}</text>\n`;
  });

  body += `<text x="40" y="36" fill="${PALETTE.text}" font-family="monospace" font-size="18" font-weight="bold">${escapeXml(title)}</text>\n`;
  body += `<text x="40" y="54" fill="${PALETTE.muted}" font-family="monospace" font-size="12">${escapeXml(subtitle)}</text>\n`;
  body += `<text x="40" y="70" fill="${PALETTE.muted}" font-family="monospace" font-size="11">All axes are latency (ms), inverted so bigger polygon = faster.</text>\n`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${totalWidth}" height="${height}" viewBox="0 0 ${totalWidth} ${height}">
  <rect width="${totalWidth}" height="${height}" fill="${PALETTE.bg}" />
  ${body}
</svg>`;
}

function prettifyModelId(raw) {
  if (!raw) return 'unknown';
  const stripped = raw
    .replace(/^onnx-community\//, '')
    .replace(/-ONNX-GQA$/i, '')
    .replace(/-ONNX$/i, '');
  const parts = stripped.split('-');
  const pretty = [];
  for (const part of parts) {
    if (/^f\d+$/i.test(part)) {
      pretty.push(part.toUpperCase());
    } else if (/^f\d+a$/i.test(part)) {
      pretty.push(part.toUpperCase());
    } else if (/^q\d+/i.test(part)) {
      pretty.push(part.toUpperCase());
    } else if (/^(bf16|fp16|fp32|int8|int4)$/i.test(part)) {
      pretty.push(part.toUpperCase());
    } else if (/^\d+[bBmM]$/.test(part)) {
      pretty.push(part.toUpperCase());
    } else if (part === 'it') {
      pretty.push('Instruct');
    } else {
      pretty.push(part.charAt(0).toUpperCase() + part.slice(1));
    }
  }
  return pretty.join(' ');
}

function prettifyWorkload(workload) {
  if (!workload) return '';
  const parts = [];
  const prefill = workload.prefillTokenTarget ?? workload.prefillTokens;
  const decode = workload.decodeTokenTarget ?? workload.decodeTokens;
  if (isFiniteNumber(prefill)) parts.push(`${prefill} prompt tokens`);
  if (isFiniteNumber(decode)) parts.push(`${decode} decode tokens`);
  const sampling = workload.sampling || {};
  if (isFiniteNumber(sampling.temperature)) {
    parts.push(sampling.temperature === 0 ? 'greedy' : `temp ${sampling.temperature}`);
  } else if (workload.id && /t0/.test(workload.id)) {
    parts.push('greedy');
  }
  if (isFiniteNumber(sampling.topK) && sampling.topK > 1) {
    parts.push(`top-k ${sampling.topK}`);
  } else if (workload.id) {
    const kMatch = workload.id.match(/k(\d+)/);
    if (kMatch && Number(kMatch[1]) > 1) parts.push(`top-k ${kMatch[1]}`);
  }
  return parts.length > 0 ? parts.join(', ') : workload.id || '';
}

function buildTitle(report, selectedSection) {
  const model = prettifyModelId(report.dopplerModelId || report.modelId);
  const targetModel = prettifyModelId(report.tjsModelId);
  return `${ENGINE_META.doppler.label} (${model}) vs ${ENGINE_META.tjs.label} (${targetModel})`;
}

function buildSubtitle(report, inputPath) {
  const timestamp = report.timestamp || 'unknown run';
  const workloadDesc = prettifyWorkload(report.workload);
  const workloadPart = workloadDesc ? ` • ${workloadDesc}` : '';
  return `${timestamp}${workloadPart}`;
}

function defaultOutputPath(inputPath, sectionLabel, chartType, width, height) {
  const parsed = path.parse(inputPath);
  const safeSection = (sectionLabel || DEFAULT_SECTION).replaceAll('/', '-');
  const filename = `${parsed.name}-${safeSection}-${chartType}-${width}x${height}.svg`;
  return path.join(parsed.dir, filename);
}

function ensureFileOutputPath(outputOption, inputPath, sectionLabel, chartType, width, height) {
  if (outputOption) return path.resolve(outputOption);
  return defaultOutputPath(inputPath, sectionLabel, chartType, width, height);
}

function main() {
  const options = parseArgs(process.argv);
  if (options.help) {
    console.log(usage());
    process.exit(0);
  }

  const entries = options.inputs.map((raw) => {
    const inputPath = path.resolve(raw);
    const report = JSON.parse(fs.readFileSync(inputPath, 'utf-8'));
    const resolved = resolveSection(report, options.section);
    if (!resolved) {
      throw new Error(`No usable benchmark section found in ${inputPath}`);
    }
    return {
      report,
      sectionPayload: resolved.payload,
      resolvedSection: resolved.section,
      inputPath,
    };
  });

  const firstEntry = entries[0];
  const title = buildTitle(firstEntry.report, firstEntry.resolvedSection);
  const isMulti = entries.length > 1;

  let svg;
  if ((options.chart === 'phases' || options.chart === 'radar') && isMulti) {
    const subtitle = firstEntry.report.timestamp || 'unknown run';

    if (options.chart === 'phases') {
      svg = renderMultiPhases(entries, options.width, title, subtitle);
    } else {
      svg = renderMultiRadar(entries, options.height, title, subtitle, options.metricIds);
    }
  } else {
    const allRows = collectRows(firstEntry.report, firstEntry.sectionPayload, options.metricIds);
    const rows = options.chart === 'radar'
      ? filterRadarRows(allRows, options.metricIds)
      : allRows;
    if (rows.length === 0 || !hasAnyValue(rows)) {
      throw new Error('No valid metric values found in selected section.');
    }

    const subtitle = buildSubtitle(firstEntry.report, firstEntry.inputPath);
    svg = options.chart === 'phases'
      ? renderPhases(rows, options.width, options.height, title, subtitle, firstEntry.resolvedSection, firstEntry.sectionPayload)
      : options.chart === 'stacked'
        ? renderStackedBars(rows, options.width, options.height, title, subtitle, firstEntry.resolvedSection)
        : options.chart === 'radar'
          ? renderRadar(rows, options.width, options.height, title, subtitle, firstEntry.resolvedSection)
          : renderBarChart(rows, options.width, options.height, title, subtitle, firstEntry.resolvedSection);
  }

  const outputPath = ensureFileOutputPath(
    options.output,
    firstEntry.inputPath,
    firstEntry.resolvedSection,
    options.chart,
    options.width,
    options.height,
  );

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, svg, 'utf-8');
  console.log(`wrote ${outputPath}`);
}

main();
