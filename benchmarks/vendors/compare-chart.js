#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  SVG_FONTS,
  SVG_THEME,
  makeSvgTextStyle,
} from './svg-theme.js';

const CHART_TYPES = Object.freeze(['bar', 'stacked', 'radar', 'phases']);
const DEFAULT_CHART = 'bar';
const DEFAULT_WIDTH = 960;
const DEFAULT_HEIGHT = 720;
const DEFAULT_SECTION = 'compute/parity';
const EMPTY_STRING = '';
const DEFAULT_UNIT = EMPTY_STRING;
const DEFAULT_SAFE_TITLE = EMPTY_STRING;
const CANVAS_PADDING = 14;
const STATIC_CHART_TITLE = 'Latency timeline on MacBook Air M3';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const COMPARE_METRICS_PATH = path.join(__dirname, 'compare-metrics.json');
const DOPPLER_HARNESS_PATH = path.join(__dirname, 'harnesses', 'doppler.json');
const TRANSFORMERSJS_HARNESS_PATH = path.join(__dirname, 'harnesses', 'transformersjs.json');
const README_SCENARIO_NAME = 'readme-evidence';

const CHART_SCENARIOS = Object.freeze({
  [README_SCENARIO_NAME]: Object.freeze({
    inputs: Object.freeze([
      path.join(__dirname, 'fixtures', 'g3-1b-p064-d064-t0-k1.compare.json'),
      path.join(__dirname, 'fixtures', 'lfm2-5-1-2b-p064-d064-t0-k1.compare.json'),
    ]),
    chart: 'phases',
    section: 'compute/parity',
    width: 1200,
    height: 474,
    output: path.join(__dirname, 'results', 'compare_1b_multi-workload_favorable_phases.svg'),
    description: 'README chart: Gemma 3 and LFM2.5 warm-opfs phase comparison',
  }),
});
const SCENARIO_NAMES = Object.keys(CHART_SCENARIOS);

const DEFAULT_METRICS = Object.freeze([
  {
    id: 'decodeTokensPerSec',
    label: 'decode tok/s',
    unit: 'tok/s',
    higherBetter: true,
  },
  {
    id: 'promptTokensPerSecToFirstToken',
    label: 'prompt tok/s to first token',
    unit: 'tok/s',
    higherBetter: true,
  },
  {
    id: 'firstTokenMs',
    label: 'first token (TTFT)',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'firstResponseMs',
    label: 'first response (first token + load)',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'decodeMs',
    label: 'decode ms',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'totalRunMs',
    label: 'total run ms',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'modelLoadMs',
    label: 'model load',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'decodeMsPerTokenP50',
    label: 'decode p50 ms/token',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'decodeMsPerTokenP95',
    label: 'decode p95 ms/token',
    unit: 'ms',
    higherBetter: false,
  },
  {
    id: 'decodeMsPerTokenP99',
    label: 'decode p99 ms/token',
    unit: 'ms',
    higherBetter: false,
  },
]);

const METRIC_PATH_HINTS = Object.freeze({
  decodeTokensPerSec: [
    'result.timing.decodeTokensPerSec',
  ],
  promptTokensPerSecToFirstToken: [
    'result.timing.promptTokensPerSecToFirstToken',
    'result.timing.prefillTokensPerSecTtft',
    'result.timing.prefillTokensPerSec',
  ],
  firstTokenMs: [
    'result.timing.firstTokenMs',
  ],
  firstResponseMs: [
    'result.timing.firstResponseMs',
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

function readJsonFile(filePath) {
  const raw = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(raw);
}

function normalizeMetricRows(payload) {
  const rows = Array.isArray(payload?.metrics) ? payload.metrics : [];
  return rows
    .filter((entry) => entry && typeof entry === 'object' && !Array.isArray(entry))
    .filter((entry) => typeof entry.id === 'string' && entry.id.trim() !== '')
    .map((entry) => ({
      id: String(entry.id),
      label: typeof entry.label === 'string' && entry.label.trim() ? entry.label : entry.id,
      unit: typeof entry.unit === 'string' ? entry.unit : '',
      higherBetter: entry.higherBetter,
      required: entry.required === true,
    }));
}

function expandMetricPathHint(pathExpr) {
  if (typeof pathExpr !== 'string' || pathExpr.trim() === '') return [];
  const trimmed = pathExpr.trim();
  if (trimmed.startsWith('result.')) {
    return [trimmed];
  }
  return [`result.${trimmed}`, trimmed];
}

function buildMetricPathHintsFromHarnesses(...harnesses) {
  const byMetric = new Map();
  for (const harness of harnesses) {
    const metricPaths = harness?.normalization?.metricPaths;
    if (!metricPaths || typeof metricPaths !== 'object') continue;
    for (const [metricId, paths] of Object.entries(metricPaths)) {
      if (!Array.isArray(paths)) continue;
      const existing = byMetric.get(metricId) || [];
      for (const candidate of paths) {
        for (const expanded of expandMetricPathHint(candidate)) {
          if (!existing.includes(expanded)) {
            existing.push(expanded);
          }
        }
      }
      byMetric.set(metricId, existing);
    }
  }
  const out = {};
  for (const [metricId, paths] of byMetric.entries()) {
    out[metricId] = paths;
  }
  return out;
}

function loadChartMetricContract() {
  const contract = readJsonFile(COMPARE_METRICS_PATH);
  const dopplerHarness = readJsonFile(DOPPLER_HARNESS_PATH);
  const transformersHarness = readJsonFile(TRANSFORMERSJS_HARNESS_PATH);
  const metrics = normalizeMetricRows(contract);
  if (metrics.length === 0) {
    throw new Error(`compare-chart: metric contract at ${COMPARE_METRICS_PATH} must define metrics.`);
  }
  const metricPathHints = buildMetricPathHintsFromHarnesses(dopplerHarness, transformersHarness);
  return Object.freeze({
    metrics,
    metricPathHints: Object.keys(metricPathHints).length > 0 ? metricPathHints : METRIC_PATH_HINTS,
  });
}

const CHART_METRIC_CONTRACT = loadChartMetricContract();
const PALETTE = SVG_THEME.palette;
const PHASE_COLORS = PALETTE.phase;
const ARCHITECTURE_COLORS = PALETTE.architecture;
const FONT_UI = SVG_FONTS.uiCss.replaceAll('"', "'");
const FONT_MONO = SVG_FONTS.monoCss.replaceAll('"', "'");
const SVG_STYLE = makeSvgTextStyle();
const PHASE_PANEL_OPACITY = '0.35';

let svgIdCounter = 0;

function renderChartCanvas(width, height) {
  return `<rect x="0" y="0" width="${width}" height="${height}" fill="#020617" />
  <rect x="${CANVAS_PADDING}" y="${CANVAS_PADDING}" width="${Math.max(0, width - CANVAS_PADDING * 2)}" height="${height - CANVAS_PADDING * 2}" rx="0" fill="#020817" fill-opacity="0.50" stroke="none" />`;
}

function renderChartHeaderBand(width, title, subtitle, sectionLabel) {
  const safeTitle = escapeXml(title || subtitle || DEFAULT_SAFE_TITLE);
  const textY = CANVAS_PADDING + 34;
  return `<text x="${CANVAS_PADDING + 16}" y="${textY}" fill="#dbeafe" font-family="${FONT_UI}" font-size="14" font-weight="bold" stroke="none">${safeTitle}</text>`;
}

function renderPhaseTrackPanel(x, y, width, height, tint = PALETTE.grid) {
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="0" fill="${PALETTE.grid}" fill-opacity="${PHASE_PANEL_OPACITY}" stroke="none" />`;
}

function renderPhaseSceneDefs() {
  return `<defs>
  <linearGradient id="phase-canvas-glow" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="#7c3aed18" />
    <stop offset="45%" stop-color="#ef444410" />
    <stop offset="100%" stop-color="#2563eb12" />
  </linearGradient>
  <linearGradient id="phase-workload-panel" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="#081121" />
    <stop offset="55%" stop-color="#0b1325" />
    <stop offset="100%" stop-color="#0c1630" />
  </linearGradient>
  <linearGradient id="phase-workload-stroke" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="${ARCHITECTURE_COLORS.loadBorder}" />
    <stop offset="48%" stop-color="${ARCHITECTURE_COLORS.edge}" />
    <stop offset="100%" stop-color="${ARCHITECTURE_COLORS.inferBorder}" />
  </linearGradient>
  <linearGradient id="phase-track-fill" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="#0c1528" />
    <stop offset="100%" stop-color="#111b31" />
  </linearGradient>
  <linearGradient id="phase-track-stroke" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="#ffffff1d" />
    <stop offset="100%" stop-color="#ffffff08" />
  </linearGradient>
  <linearGradient id="phase-load-grad" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="#f87171" />
    <stop offset="100%" stop-color="${PHASE_COLORS.warmLoad}" />
  </linearGradient>
  <linearGradient id="phase-prefill-grad" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="#a855f7" />
    <stop offset="100%" stop-color="${ARCHITECTURE_COLORS.edge}" />
  </linearGradient>
  <linearGradient id="phase-decode-grad" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="#60a5fa" />
    <stop offset="100%" stop-color="${PHASE_COLORS.decode}" />
  </linearGradient>
  <linearGradient id="phase-doppler-chip" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="#7c3aed33" />
    <stop offset="100%" stop-color="#ef444422" />
  </linearGradient>
  <linearGradient id="phase-transformers-chip" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="#1d4ed833" />
    <stop offset="100%" stop-color="#ffbd4522" />
  </linearGradient>
  <linearGradient id="phase-total-pill" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="#0f172a" />
    <stop offset="100%" stop-color="#172554" />
  </linearGradient>
  <linearGradient id="phase-winner-pill" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="#7c3aed33" />
    <stop offset="100%" stop-color="#2563eb33" />
  </linearGradient>
  </defs>`;
}

function svgWrap(width, height, body, title = 'Benchmark Comparison', desc = '') {
  svgIdCounter += 1;
  const titleId = `chart-title-${svgIdCounter}`;
  const descId = `chart-desc-${svgIdCounter}`;
  const safeTitle = escapeXml(title);
  const safeDesc = escapeXml(desc || 'Benchmark comparison chart');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${titleId} ${descId}">
  <title id="${titleId}">${safeTitle}</title>
  <desc id="${descId}">${safeDesc}</desc>
  ${SVG_STYLE}
  ${renderChartCanvas(width, height)}
  ${body}
</svg>`;
}

const ENGINE_META = Object.freeze({
  doppler: {
    key: 'doppler',
    label: 'Doppler.js',
    color: PALETTE.doppler,
  },
  transformersjs: {
    key: 'transformersjs',
    label: 'Transformers.js (v4)',
    color: PALETTE.transformersjs,
  },
});

const DERIVED_KEY = Object.freeze({
  doppler: 'doppler',
  transformersjs: 'transformersjs',
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
    `  --scenario <${SCENARIO_NAMES.join(', ')}>  Use predefined chart scenario (main chart generator shortcut)`,
    `  --scenario ${README_SCENARIO_NAME}   Gemma 3 + LFM2.5 warm-opfs phase evidence chart`,
    '  --include-workload <id|label>  Include only workloads matching by id or rendered label (repeatable)',
    '  --exclude-workload <id|label>  Exclude workloads by id or rendered label (repeatable)',
    '  --allow-non-comparable        Allow mixed benchmark settings across inputs (default: strict apples-to-apples)',
    '  --metrics <id,id,...>         Comma-separated metric IDs',
    '  --width <n>                   SVG width (default: 960)',
    '  --height <n>                  SVG height (default: 560)',
    '  --help                        Show this help text',
    '',
    'Examples:',
    '  node benchmarks/vendors/compare-chart.js --input benchmarks/vendors/fixtures/sample-compare.json',
    '  node benchmarks/vendors/compare-chart.js --input ... --chart stacked --width 1200',
    '  node benchmarks/vendors/compare-chart.js --input ... --chart radar --metrics decodeTokensPerSec,firstTokenMs',
    '  node benchmarks/vendors/compare-chart.js --input ... --chart phases --include-workload "64 prompt tokens, 64 decode tokens, greedy"',
    '  node benchmarks/vendors/compare-chart.js --input ... --chart phases --exclude-workload p064-d064-t0-k1',
    '  node benchmarks/vendors/compare-chart.js --chart phases --input workload1.json --input workload2.json',
    `  node benchmarks/vendors/compare-chart.js --scenario ${README_SCENARIO_NAME}`,
    '  node benchmarks/vendors/compare-chart.js --chart radar --input workload1.json --input workload2.json',
  ].join('\n');
}

function splitCommaList(raw) {
  return String(raw || EMPTY_STRING)
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

function splitWorkloadFilterList(raw) {
  const rawText = String(raw || EMPTY_STRING).trim();
  if (rawText.length === 0) return [];
  const parts = rawText.split(',').map((item) => item.trim()).filter((item) => item.length > 0);
  if (parts.length === 0) return [];
  if (parts.length === 1) return [rawText];

  const isLikelyFilterList = parts.every((item) => !/\s/.test(item));
  if (!isLikelyFilterList) return [rawText];

  return parts;
}

function normalizeWorkloadFilterToken(raw) {
  return String(raw || EMPTY_STRING)
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ');
}

function workloadFilterTokens(entry) {
  const workload = entry?.report?.workload || {};
  const inputBase = path.basename(entry.inputPath || EMPTY_STRING, '.json');
  const sampling = workload.sampling || {};
  const prefill = workload.prefillTokenTarget ?? workload.prefillTokens;
  const decode = workload.decodeTokenTarget ?? workload.decodeTokens;
  const label = prettifyWorkload(workload);

  const tokens = new Set([
    inputBase,
    workload.id,
    label,
    `${prefill} ${decode}`,
    `${prefill} prompt tokens, ${decode} decode tokens`,
  ]);

  if (isFiniteNumber(sampling.temperature)) {
    tokens.add(`temperature ${sampling.temperature}`);
    if (sampling.temperature === 0) {
      tokens.add('greedy');
      tokens.add(`${prefill} prompt tokens, ${decode} decode tokens, greedy`);
    } else {
      tokens.add(`temp ${sampling.temperature}`);
    }
  }
  if (isFiniteNumber(sampling.topK)) {
    tokens.add(`k${sampling.topK}`);
    tokens.add(`top-k ${sampling.topK}`);
    if (sampling.topK > 1) {
      tokens.add(`${prefill} prompt tokens, ${decode} decode tokens, top-k ${sampling.topK}`);
    }
  }

  const normalized = new Set();
  for (const token of tokens) {
    const normalizedToken = normalizeWorkloadFilterToken(token);
    if (normalizedToken.length > 0) normalized.add(normalizedToken);
  }
  return normalized;
}

function shouldExcludeWorkload(entry, excludeSet) {
  if (excludeSet.size === 0) return false;
  const tokens = workloadFilterTokens(entry);
  const workloadLabel = normalizeWorkloadFilterToken(prettifyWorkload(entry.report?.workload));
  for (const selector of excludeSet) {
    if (tokens.has(selector)) return true;
    if (workloadLabel.length > 0 && workloadLabel.includes(selector)) return true;
  }
  return false;
}

function shouldIncludeWorkload(entry, includeSet) {
  if (includeSet.size === 0) return true;
  const tokens = workloadFilterTokens(entry);
  const workloadLabel = normalizeWorkloadFilterToken(prettifyWorkload(entry.report?.workload));
  for (const selector of includeSet) {
    if (tokens.has(selector)) return true;
    if (workloadLabel.length > 0 && workloadLabel.includes(selector)) return true;
  }
  return false;
}

function parseArgs(argv) {
  const parsed = {
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT,
    chart: DEFAULT_CHART,
    section: DEFAULT_SECTION,
    sectionExplicit: false,
    chartExplicit: false,
    widthExplicit: false,
    heightExplicit: false,
    outputExplicit: false,
    scenario: null,
    metricIds: [],
    includeWorkloads: [],
    excludeWorkloads: [],
    allowNonComparable: false,
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
      parsed.outputExplicit = true;
      continue;
    }
    if (arg.startsWith('--output=')) {
      parsed.output = arg.substring('--output='.length);
      parsed.outputExplicit = true;
      continue;
    }

    if (arg === '--section') {
      parsed.section = argv[i + 1] || DEFAULT_SECTION;
      parsed.sectionExplicit = true;
      i += 1;
      continue;
    }
    if (arg.startsWith('--section=')) {
      parsed.section = arg.substring('--section='.length);
      parsed.sectionExplicit = true;
      continue;
    }

    if (arg === '--chart') {
      parsed.chart = argv[i + 1];
      parsed.chartExplicit = true;
      i += 1;
      continue;
    }
    if (arg.startsWith('--chart=')) {
      parsed.chart = arg.substring('--chart='.length);
      parsed.chartExplicit = true;
      continue;
    }

    if (arg === '--metrics') {
      parsed.metricIds = String(argv[i + 1] || EMPTY_STRING)
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

    if (arg === '--exclude-workload') {
      parsed.excludeWorkloads.push(...splitWorkloadFilterList(argv[i + 1]));
      i += 1;
      continue;
    }
    if (arg.startsWith('--exclude-workload=')) {
      parsed.excludeWorkloads.push(...splitWorkloadFilterList(arg.substring('--exclude-workload='.length)));
      continue;
    }

    if (arg === '--include-workload') {
      parsed.includeWorkloads.push(...splitWorkloadFilterList(argv[i + 1]));
      i += 1;
      continue;
    }
    if (arg.startsWith('--include-workload=')) {
      parsed.includeWorkloads.push(...splitWorkloadFilterList(arg.substring('--include-workload='.length)));
      continue;
    }

    if (arg === '--width') {
      parsed.width = Number.parseInt(argv[i + 1], 10);
      parsed.widthExplicit = true;
      i += 1;
      continue;
    }
    if (arg.startsWith('--width=')) {
      parsed.width = Number.parseInt(arg.substring('--width='.length), 10);
      parsed.widthExplicit = true;
      continue;
    }

    if (arg === '--height') {
      parsed.height = Number.parseInt(argv[i + 1], 10);
      parsed.heightExplicit = true;
      i += 1;
      continue;
    }
    if (arg.startsWith('--height=')) {
      parsed.height = Number.parseInt(arg.substring('--height='.length), 10);
      parsed.heightExplicit = true;
      continue;
    }

    if (arg === '--allow-non-comparable') {
      parsed.allowNonComparable = true;
      continue;
    }

    if (arg === '--scenario') {
      parsed.scenario = (argv[i + 1] || EMPTY_STRING).trim();
      i += 1;
      continue;
    }
    if (arg.startsWith('--scenario=')) {
      parsed.scenario = arg.substring('--scenario='.length).trim();
      continue;
    }

    if (arg.startsWith('--')) {
      throw new Error(`Unknown argument: ${arg}`);
    }
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

  parsed.excludeWorkloads = [...new Set(parsed.excludeWorkloads.map((item) => item.trim()).filter((item) => item.length > 0))];
  parsed.includeWorkloads = [...new Set(parsed.includeWorkloads.map((item) => item.trim()).filter((item) => item.length > 0))];
  return parsed;
}

function normalizeComparableNumber(value) {
  if (!isFiniteNumber(value)) return null;
  return Number(value.toFixed(6));
}

function normalizeComparableValue(value) {
  if (isFiniteNumber(value)) return normalizeComparableNumber(value);
  if (value == null) return null;
  return value;
}

function firstNonNull(...values) {
  for (const value of values) {
    if (value != null) return value;
  }
  return null;
}

function normalizeComparableSectionLabel(sectionLabel) {
  const normalized = String(sectionLabel || EMPTY_STRING).trim();
  if (normalized === 'warm' || normalized === 'compute/parity') return 'warm-parity';
  if (normalized === 'compute/throughput') return 'warm-throughput';
  return normalized.length > 0 ? normalized : null;
}

function normalizeComparableModeLabel(modeLabel, comparableSection) {
  if (comparableSection === 'warm-parity' || comparableSection === 'warm-throughput') {
    return 'warm';
  }
  return modeLabel == null ? null : String(modeLabel);
}

function buildComparabilityRecord(entry) {
  const report = entry?.report || {};
  const workload = report.workload || {};
  const sampling = workload.sampling || {};
  const sectionPayload = entry?.sectionPayload || {};
  const dopplerRequest = sectionPayload?.doppler?.request || {};
  const tjsPayload = sectionPayload?.transformersjs ?? sectionPayload?.tjs ?? {};
  const resolvedMode = typeof entry?.resolvedSection === 'string'
    ? entry.resolvedSection.split('/')[0]
    : null;
  const comparableSection = normalizeComparableSectionLabel(entry?.resolvedSection || null);
  const comparableMode = normalizeComparableModeLabel(firstNonNull(report.mode, resolvedMode), comparableSection);

  return {
    section: normalizeComparableValue(comparableSection),
    mode: normalizeComparableValue(comparableMode),
    decodeProfile: normalizeComparableValue(report.decodeProfile ?? null),
    workloadId: normalizeComparableValue(workload.id ?? null),
    prefillTokenTarget: normalizeComparableValue(firstNonNull(
      workload.prefillTokenTarget,
      workload.prefillTokens,
    )),
    decodeTokenTarget: normalizeComparableValue(firstNonNull(
      workload.decodeTokenTarget,
      workload.decodeTokens,
    )),
    temperature: normalizeComparableValue(sampling.temperature ?? null),
    topK: normalizeComparableValue(sampling.topK ?? null),
    topP: normalizeComparableValue(sampling.topP ?? null),
    warmupRuns: normalizeComparableValue(report.warmupRuns ?? null),
    runs: normalizeComparableValue(report.runs ?? null),
    seed: normalizeComparableValue(report.seed ?? null),
    loadMode: normalizeComparableValue(firstNonNull(
      report.methodology?.loadMode,
      dopplerRequest.loadMode,
      tjsPayload.loadMode,
      report.loadMode,
    )),
    cacheMode: normalizeComparableValue(firstNonNull(
      dopplerRequest.cacheMode,
      report.cacheMode,
    )),
  };
}

function formatComparableValue(value) {
  if (value == null) return 'null';
  if (typeof value === 'string') return value;
  return String(value);
}

function assertComparableEntries(entries, { allowNonComparable = false } = {}) {
  if (allowNonComparable || !Array.isArray(entries) || entries.length <= 1) {
    return;
  }

  const records = entries.map((entry) => ({
    inputPath: entry.inputPath,
    values: buildComparabilityRecord(entry),
  }));
  const fieldOrder = Object.keys(records[0].values);
  const mismatchedFields = [];

  for (const field of fieldOrder) {
    const baseline = records[0].values[field];
    const mismatch = records.some((record, index) => index > 0 && record.values[field] !== baseline);
    if (mismatch) {
      mismatchedFields.push(field);
    }
  }

  if (mismatchedFields.length === 0) {
    return;
  }

  const fieldLabels = Object.freeze({
    section: 'section',
    mode: 'mode',
    decodeProfile: 'decode profile',
    workloadId: 'workload id',
    prefillTokenTarget: 'prefill token target',
    decodeTokenTarget: 'decode token target',
    temperature: 'temperature',
    topK: 'top-k',
    topP: 'top-p',
    warmupRuns: 'warmup runs',
    runs: 'timed runs',
    seed: 'seed',
    loadMode: 'load mode',
    cacheMode: 'cache mode',
  });

  const detailLines = records.map((record) => {
    const fileLabel = path.basename(record.inputPath);
    const parts = mismatchedFields.map((field) => {
      const label = fieldLabels[field] || field;
      return `${label}=${formatComparableValue(record.values[field])}`;
    });
    return `- ${fileLabel}: ${parts.join(', ')}`;
  });

  throw new Error([
    'Inputs are not apples-to-apples for charting.',
    `Mismatched fields: ${mismatchedFields.map((field) => fieldLabels[field] || field).join(', ')}`,
    ...detailLines,
    'Re-run with matched benchmark settings or pass --allow-non-comparable to bypass.',
  ].join('\n'));
}

function applyScenarioOptions(parsed) {
  if (!parsed.scenario) return;
  const scenario = CHART_SCENARIOS[parsed.scenario];
  if (!scenario) {
    const names = SCENARIO_NAMES.length > 0 ? SCENARIO_NAMES.join(', ') : 'none';
    throw new Error(`Unknown scenario "${parsed.scenario}". Available scenarios: ${names}`);
  }

  if (parsed.inputs.length > 0) {
    throw new Error('--scenario is not compatible with --input; use one or the other');
  }

  parsed.inputs = [...scenario.inputs];
  if (!parsed.chartExplicit && scenario.chart) parsed.chart = scenario.chart;
  if (!parsed.sectionExplicit && scenario.section) parsed.section = scenario.section;
  if (!parsed.widthExplicit && scenario.width) parsed.width = scenario.width;
  if (!parsed.heightExplicit && scenario.height) parsed.height = scenario.height;
  if (!parsed.outputExplicit && scenario.output) {
    parsed.output = scenario.output;
  }
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
  return Boolean(section?.doppler || section?.transformersjs || section?.result || section?.tjs);
}

function resolveSection(report, requestedSection) {
  if (!report || typeof report !== 'object') return null;
  if (!report.sections) {
    return sectionHasComparable(report)
      ? { section: requestedSection, payload: report }
      : null;
  }

  let cursor = report.sections;
  const segments = String(requestedSection || EMPTY_STRING)
    .split('/')
    .map((segment) => segment.trim())
    .filter((segment) => segment.length > 0);
  for (const segment of segments) {
    if (!cursor || typeof cursor !== 'object') {
      cursor = null;
      break;
    }
    cursor = cursor[segment];
  }
  if (sectionHasComparable(cursor)) {
    return { section: requestedSection, payload: cursor };
  }

  const normalizedSection = String(requestedSection || EMPTY_STRING).trim();
  if (normalizedSection === 'compute/parity' || normalizedSection === 'compute/throughput') {
    const warmPayload = report.sections.warm;
    if (sectionHasComparable(warmPayload)) {
      return { section: 'warm', payload: warmPayload };
    }
  }
  if (normalizedSection === 'warm') {
    const parityPayload = report.sections.compute?.parity;
    if (sectionHasComparable(parityPayload)) {
      return { section: 'compute/parity', payload: parityPayload };
    }
  }
  return null;
}

function enginePayload(sectionPayload, engineId) {
  if (!sectionPayload || typeof sectionPayload !== 'object') return null;
  if (engineId === 'transformersjs') {
    return sectionPayload.transformersjs ?? sectionPayload.tjs ?? null;
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

  const hintPaths = CHART_METRIC_CONTRACT.metricPathHints[metricDef.id] || METRIC_PATH_HINTS[metricDef.id] || [];
  const value = getFirstFinitePathValue(scope, hintPaths);
  if (value !== null) return { value, status: 'ok' };

  const direct = scope.result?.[metricDef.id] ?? scope[metricDef.id];
  const directValue = toNumber(direct);
  return { value: directValue, status: directValue === null ? 'missing' : 'ok' };
}

function metricRowsFromReport(report, sectionPayload, metricIds) {
  const sourceMetrics = Array.isArray(report.metricContract?.metrics) && report.metricContract.metrics.length > 0
    ? report.metricContract.metrics
    : CHART_METRIC_CONTRACT.metrics;
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
  const tjsPayload = enginePayload(sectionPayload, 'transformersjs');

  return metrics.map((metricDef) => {
    if (typeof metricDef.higherBetter !== 'boolean') {
      throw new Error(`Metric "${metricDef.id}" is missing required boolean "higherBetter".`);
    }
    const doppler = resolveMetric(dopplerPayload, metricDef, 'doppler');
    const transformersjs = resolveMetric(tjsPayload, metricDef, 'transformersjs');
    return {
      id: metricDef.id,
      label: metricDef.label || metricDef.id,
      unit: metricDef.unit || DEFAULT_UNIT,
      higherBetter: metricDef.higherBetter,
      doppler,
      transformersjs,
    };
  });
}

function hasAnyValue(rows) {
  return rows.some((row) => isFiniteNumber(row.doppler.value) || isFiniteNumber(row.transformersjs.value));
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
    const values = [row.doppler.value, row.transformersjs.value].filter((value) => isFiniteNumber(value));
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
    const tjsScore = scoreMetric(row, scales.get(row.id), 'transformersjs');
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
      metric.transformersjs.value ?? 1,
      ...(isFiniteNumber(metric.doppler.value) ? [metric.doppler.value] : []),
      ...(isFiniteNumber(metric.transformersjs.value) ? [metric.transformersjs.value] : []),
    );
    const dopplerWidth = isFiniteNumber(metric.doppler.value) ? (metric.doppler.value / maxValue) * barAreaMax : 0;
    const tjsWidth = isFiniteNumber(metric.transformersjs.value) ? (metric.transformersjs.value / maxValue) * barAreaMax : 0;
    const rowCenter = y + 34;

    body += `<text x="32" y="${y + 6}" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="13">${escapeXml(metric.label)}</text>\n`;
    body += `<text x="32" y="${y + 22}" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="11">${metric.higherBetter ? 'higher is better' : 'lower is better'}</text>\n`;
    body += `<line x1="${left}" y1="${y + 44}" x2="${left + barAreaMax}" y2="${y + 44}" stroke="${PALETTE.grid}" stroke-width="1" />\n`;

    if (metric.doppler.status === 'ok') {
      body += `<rect x="${left + 80}" y="${rowCenter - 18}" width="${dopplerWidth}" height="16" fill="${PALETTE.doppler}" />\n`;
      body += `<text x="${left + 88 + dopplerWidth}" y="${rowCenter - 5}" fill="${PALETTE.doppler}" font-family="${FONT_MONO}" font-size="11">${formatValue(metric.doppler.value, metric.unit)}</text>\n`;
      if (!isFiniteNumber(metric.doppler.value)) {
        body += `<text x="${left + 90}" y="${rowCenter - 5}" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="11">n/a</text>\n`;
      }
    } else {
      body += `<rect x="${left + 80}" y="${rowCenter - 18}" width="${barAreaMax}" height="16" fill="${PALETTE.failFill}" />\n`;
      body += `<text x="${left + 90}" y="${rowCenter - 5}" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="11">Doppler.js failed</text>\n`;
    }

    if (metric.transformersjs.status === 'ok') {
      body += `<rect x="${left + 80}" y="${rowCenter + 2}" width="${tjsWidth}" height="16" fill="${PALETTE.transformersjs}" />\n`;
      body += `<text x="${left + 88 + tjsWidth}" y="${rowCenter + 15}" fill="${PALETTE.transformersjs}" font-family="${FONT_MONO}" font-size="11">${formatValue(metric.transformersjs.value, metric.unit)}</text>\n`;
      if (!isFiniteNumber(metric.transformersjs.value)) {
        body += `<text x="${left + 90}" y="${rowCenter + 15}" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="11">n/a</text>\n`;
      }
    } else {
      body += `<rect x="${left + 80}" y="${rowCenter + 2}" width="${barAreaMax}" height="16" fill="${PALETTE.failFill}" />\n`;
      body += `<text x="${left + 90}" y="${rowCenter + 15}" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="11">Transformers.js failed</text>\n`;
    }
  });

  body += `<text x="${left + 80}" y="${top - 24}" fill="${PALETTE.doppler}" stroke="#ffffff" stroke-width="2" font-family="${FONT_MONO}" font-size="14">${ENGINE_META.doppler.label}</text>\n`;
  body += `<text x="${left + 170}" y="${top - 24}" fill="${PALETTE.transformersjs}" stroke="#ffffff" stroke-width="2" font-family="${FONT_MONO}" font-size="14">${ENGINE_META.transformersjs.label}</text>\n`;
  body += `<text x="40" y="36" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="18" font-weight="bold">${escapeXml(title)}</text>\n`;
  body += `<text x="40" y="54" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="12">${escapeXml(subtitle)}</text>\n`;
  body += `<text x="40" y="${height - 22}" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="11">${escapeXml(`Section: ${sectionLabel}`)}</text>\n`;

  return svgWrap(width, height, body, title, `${subtitle} • Section: ${sectionLabel}`);
}

function renderStackedBars(rows, width, height, title, subtitle, sectionLabel) {
  const chartLeft = 220;
  const chartRight = 44;
  const barMaxWidth = width - chartLeft - chartRight;
  const baseY = 102;
  const rowHeight = 110;
  const top = 84;
  const engines = [ENGINE_META.doppler, ENGINE_META.transformersjs];
  const scaledRows = buildScaledRows(rows);
  let body = '';
  const scale = resolveScale(rows);

  engines.forEach((engine, index) => {
    const y = top + index * rowHeight;
    const label = `${engine.label} (normalized metric composition)`;
    const scoreTotal = scaledRows.reduce((sum, metric) => sum + (scoreMetric(metric, scale.get(metric.id), engine.key)), 0);
    const base = baseY + index * rowHeight;

    body += `<text x="32" y="${y + 8}" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="14" font-weight="bold">${escapeXml(label)}</text>\n`;
    body += `<line x1="${chartLeft}" y1="${base + 26}" x2="${chartLeft + barMaxWidth}" y2="${base + 26}" stroke="${PALETTE.grid}" stroke-width="1" />\n`;

    if (scoreTotal <= 0) {
      body += `<text x="${chartLeft + 90}" y="${base + 18}" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="11">${engine.label} has no valid metric values.</text>\n`;
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
        body += `<text x="${cursor + 4}" y="${base + 23}" fill="#000" font-family="${FONT_MONO}" font-size="10">${safeLabel}</text>\n`;
      }
      cursor += segmentWidth;
    });
  });

  rows.forEach((metric, index) => {
    const x = 32 + (index % 2) * ((width - 64) / 2);
    const y = height - 50 + Math.floor(index / 2) * 16;
    const fill = PALETTE.metric[index % PALETTE.metric.length];
    body += `<rect x="${x}" y="${y}" width="14" height="14" fill="${fill}" />\n`;
    body += `<text x="${x + 18}" y="${y + 11}" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="11">${escapeXml(metric.label)}</text>\n`;
  });

  body += `<text x="40" y="36" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="18" font-weight="bold">${escapeXml(title)}</text>\n`;
  body += `<text x="40" y="54" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="12">${escapeXml(subtitle)}</text>\n`;
  body += `<text x="40" y="${height - 22}" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="11">${escapeXml(`Section: ${sectionLabel}`)}</text>\n`;

  return svgWrap(width, height, body, title, `${subtitle} • Section: ${sectionLabel}`);
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
  const radius = Math.max(40, Math.min(width, height - 170) / 2 - 110);
  const ringCount = 4;
  const axisCount = Math.max(rows.length, 1);
  const angleStep = (Math.PI * 2) / axisCount;
  const scale = resolveScale(rows);
  const scaledRows = buildScaledRows(rows);
  let body = '';

  if (rows.length < 3) {
    body += `<text x="40" y="36" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="18" font-weight="bold">${escapeXml(title)}</text>\n`;
    body += `<text x="40" y="54" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="12">${escapeXml(subtitle)}</text>\n`;
    body += `<text x="${centerX}" y="${centerY}" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="12" text-anchor="middle">Radar chart requires at least 3 metrics.</text>\n`;
    body += `<text x="40" y="${height - 22}" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="11">${escapeXml(`Section: ${sectionLabel}`)}</text>\n`;
    return svgWrap(width, height, body, title, `${subtitle} • Section: ${sectionLabel}`);
  }

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
    body += `<text x="${x}" y="${y}" fill="${PALETTE.text}" text-anchor="${anchor}" font-family="${FONT_MONO}" font-size="11">${escapeXml(metric.label)}</text>\n`;
  });

  const engines = [ENGINE_META.doppler, ENGINE_META.transformersjs];
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
    body += `<text x="${width - 170}" y="${yLegend + 1}" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="12">${engine.label}</text>\n`;
  });

  const allBad = engines.every((engine) => {
    const metricScores = rows.map((metric) => {
      return engine.key === 'doppler'
        ? scoreMetric(metric, scale.get(metric.id), 'doppler')
        : scoreMetric(metric, scale.get(metric.id), 'transformersjs');
    });
    return metricScores.every((value) => value <= 0);
  });
  if (allBad) {
    body += `<text x="${centerX - 120}" y="${centerY}" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="12">No comparable valid values for this section.</text>\n`;
  }

  body += `<text x="40" y="36" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="18" font-weight="bold">${escapeXml(title)}</text>\n`;
  body += `<text x="40" y="54" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="12">${escapeXml(subtitle)}</text>\n`;
  body += `<text x="40" y="70" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="11">All axes are latency (ms), inverted so bigger polygon = faster.</text>\n`;
  body += `<text x="40" y="${height - 22}" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="11">${escapeXml(`Section: ${sectionLabel}`)}</text>\n`;

  return svgWrap(width, height, body, title, `${subtitle} • Section: ${sectionLabel}`);
}

const LOAD_LABEL = Object.freeze({
  doppler: 'OPFS \u2192 VRAM',
  transformersjs: 'OPFS \u2192 ORT \u2192 VRAM',
});

function resolvePhaseValues(sectionPayload, engineId) {
  const payload = enginePayload(sectionPayload, engineId);
  if (!payload || payload.failed === true) return null;
  const scope = payload.result ? payload : { result: payload };
  const timing = scope.result?.timing || scope.result || {};
  const metrics = scope.result?.metrics || {};

  const modelLoadMs = toNumber(timing.modelLoadMs);
  const prefillMs = toNumber(timing.prefillMs);
  const firstTokenMs = toNumber(timing.firstTokenMs);
  const decodeMs = toNumber(timing.decodeMs);
  const promptTokensPerSecToFirstToken = firstFinite([
    timing.promptTokensPerSecToFirstToken,
    metrics.promptTokensPerSecToFirstToken,
    metrics.medianPrefillTokensPerSecTtft,
    metrics.avgPrefillTokensPerSecTtft,
    timing.prefillTokensPerSecTtft,
    metrics.prefillTokensPerSecTtft,
    timing.prefillTokensPerSec,
    metrics.prefillTokensPerSec,
  ]);
  const decodeTokensPerSec = firstFinite([
    timing.decodeTokensPerSec,
    metrics.decodeTokensPerSec,
  ]);

  const ttft = isFiniteNumber(firstTokenMs) ? firstTokenMs : prefillMs;

  const endToEnd = [modelLoadMs, ttft, decodeMs]
    .filter((v) => isFiniteNumber(v))
    .reduce((sum, v) => sum + v, 0);

  return {
    modelLoadMs,
    ttft,
    prefillMs,
    promptTokensPerSecToFirstToken,
    decodeMs,
    decodeTokensPerSec,
    endToEnd,
  };
}

function formatDurationCompact(value) {
  if (!isFiniteNumber(value)) return 'n/a';
  if (value >= 1000) {
    return `${(value / 1000).toFixed(value >= 10000 ? 1 : 2)} s`;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ms`;
}

function formatThroughputCompact(value) {
  if (!isFiniteNumber(value)) return 'n/a';
  const decimals = value >= 100 ? 1 : 2;
  return `${value.toFixed(decimals)} tok/s`;
}

function computePhaseRaceSummary(dopplerPhases, transformersPhases) {
  if (!dopplerPhases || !transformersPhases) return null;
  if (!isFiniteNumber(dopplerPhases.endToEnd) || !isFiniteNumber(transformersPhases.endToEnd)) return null;
  if (dopplerPhases.endToEnd === transformersPhases.endToEnd) {
    return { winner: null, deltaPct: 0 };
  }
  const winner = dopplerPhases.endToEnd < transformersPhases.endToEnd
    ? ENGINE_META.doppler
    : ENGINE_META.transformersjs;
  const faster = winner.key === ENGINE_META.doppler.key ? dopplerPhases : transformersPhases;
  const slower = winner.key === ENGINE_META.doppler.key ? transformersPhases : dopplerPhases;
  const deltaPct = slower.endToEnd > 0 ? ((slower.endToEnd - faster.endToEnd) / slower.endToEnd) * 100 : 0;
  return { winner, deltaPct };
}

function formatPhaseSegmentValue(segmentTitle, resolved) {
  if (!resolved) return 'n/a';
  if (segmentTitle === 'First token') {
    return `${formatDurationCompact(resolved.ttft)} • ${formatThroughputCompact(resolved.promptTokensPerSecToFirstToken)}`;
  }
  if (segmentTitle === 'Decode') {
    return `${formatDurationCompact(resolved.decodeMs)} • ${formatThroughputCompact(resolved.decodeTokensPerSec)}`;
  }
  return formatDurationCompact(resolved.modelLoadMs);
}

function renderPhaseSegmentLabel(x, y, width, title, valueLabel) {
  if (width < 52) {
    return `<text x="${x + Math.max(width / 2, 8)}" y="${y - 6}" text-anchor="middle" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="9">${escapeXml(valueLabel)}</text>\n`;
  }
  if (width < 92) {
    return `<text x="${x + 8}" y="${y + 30}" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="10">${escapeXml(valueLabel)}</text>\n`;
  }
  return `<text x="${x + 10}" y="${y + 24}" fill="${PALETTE.text}" font-family="${FONT_UI}" font-size="11" font-weight="bold">${escapeXml(title)}</text>\n`
    + `<text x="${x + 10}" y="${y + 38}" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="10">${escapeXml(valueLabel)}</text>\n`;
}

function renderEngineChip(x, y, engine) {
  const chipId = engine.key === ENGINE_META.doppler.key ? 'phase-doppler-chip' : 'phase-transformers-chip';
  const chipWidth = engine.key === ENGINE_META.doppler.key ? 128 : 188;
  return `<rect x="${x}" y="${y}" width="${chipWidth}" height="28" rx="14" fill="url(#${chipId})" stroke="${engine.color}" stroke-width="1.5" />\n`
    + `<text x="${x + 14}" y="${y + 19}" fill="${engine.color}" font-family="${FONT_UI}" font-size="14" font-weight="bold" stroke="none">${escapeXml(engine.label)}</text>\n`;
}

function renderPhaseLane({ x, y, width, engine, resolved, globalMax }) {
  const labelWidth = 184;
  const totalWidth = 118;
  const gap = 18;
  const barHeight = 56;
  const barX = x + labelWidth + gap;
  const barWidth = width - labelWidth - totalWidth - gap * 2;

  let body = '';
  body += renderEngineChip(x, y + 12, engine);
  body += `<text x="${x + 14}" y="${y + 54}" fill="${PALETTE.muted}" font-family="${FONT_UI}" font-size="12" stroke="none">${escapeXml(LOAD_LABEL[engine.key] || 'Model load')}</text>\n`;
  body += `<rect x="${barX}" y="${y}" width="${barWidth}" height="${barHeight}" rx="18" fill="url(#phase-track-fill)" stroke="url(#phase-track-stroke)" stroke-width="1.2" />\n`;

  if (!resolved) {
    body += `<rect x="${barX + 4}" y="${y + 4}" width="${Math.max(0, barWidth - 8)}" height="${barHeight - 8}" rx="14" fill="${PALETTE.failFill}" />\n`;
    body += `<text x="${barX + 18}" y="${y + 34}" fill="${PALETTE.text}" font-family="${FONT_UI}" font-size="13" font-weight="bold">No data</text>\n`;
  } else {
    const pxPerMs = Math.max(0, (barWidth - 8) / globalMax);
    let cursor = barX + 4;
    const segments = [
      { title: 'Warm load', fill: 'url(#phase-load-grad)', value: resolved.modelLoadMs },
      { title: 'First token', fill: 'url(#phase-prefill-grad)', value: resolved.ttft },
      { title: 'Decode', fill: 'url(#phase-decode-grad)', value: resolved.decodeMs },
    ];

    for (const segment of segments) {
      if (!isFiniteNumber(segment.value) || segment.value <= 0) continue;
      const segmentWidth = Math.max(2, segment.value * pxPerMs);
      body += `<rect x="${cursor}" y="${y + 4}" width="${segmentWidth}" height="${barHeight - 8}" fill="${segment.fill}" />\n`;
      body += renderPhaseSegmentLabel(cursor, y + 4, segmentWidth, segment.title, formatPhaseSegmentValue(segment.title, resolved));
      cursor += segmentWidth;
    }
  }

  const totalX = x + width - totalWidth;
  body += `<rect x="${totalX}" y="${y + 9}" width="${totalWidth}" height="38" rx="19" fill="url(#phase-total-pill)" stroke="#ffffff1f" stroke-width="1.1" />\n`;
  body += `<text x="${totalX + totalWidth / 2}" y="${y + 25}" text-anchor="middle" fill="${PALETTE.muted}" font-family="${FONT_UI}" font-size="10" font-weight="bold" stroke="none">TOTAL</text>\n`;
  body += `<text x="${totalX + totalWidth / 2}" y="${y + 39}" text-anchor="middle" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="13">${escapeXml(formatDurationCompact(resolved?.endToEnd ?? null))}</text>\n`;
  return body;
}

function renderPhaseWorkloadPanel({ x, y, width, workloadLabel, dopplerPhases, transformersPhases, globalMax }) {
  const headerHeight = 54;
  const laneGap = 16;
  const laneHeight = 56;
  const laneX = x + 24;
  const laneWidth = width - 48;
  const race = computePhaseRaceSummary(dopplerPhases, transformersPhases);
  const panelHeight = 190;

  let body = '';
  body += `<rect x="${x}" y="${y}" width="${width}" height="${panelHeight}" rx="26" fill="url(#phase-workload-panel)" stroke="url(#phase-workload-stroke)" stroke-width="1.4" />\n`;
  body += `<line x1="${x + 24}" y1="${y + 46}" x2="${x + width - 24}" y2="${y + 46}" stroke="#ffffff14" stroke-width="1" />\n`;
  body += `<text x="${x + 26}" y="${y + 32}" fill="${PALETTE.text}" font-family="${FONT_UI}" font-size="20" font-weight="bold" stroke="none">${escapeXml(workloadLabel)}</text>\n`;

  if (race?.winner) {
    const pillWidth = 220;
    const pillX = x + width - pillWidth - 24;
    const winnerShort = race.winner.key === ENGINE_META.doppler.key ? 'Doppler' : 'TJS v4';
    body += `<rect x="${pillX}" y="${y + 18}" width="${pillWidth}" height="28" rx="14" fill="#22c55e" stroke="none" />\n`;
    body += `<text x="${pillX + pillWidth / 2}" y="${y + 37}" text-anchor="middle" fill="#ffffff" font-family="${FONT_UI}" font-size="18" font-weight="bold" stroke="none">${escapeXml(`${winnerShort} ${race.deltaPct.toFixed(1)}% faster`)}</text>\n`;
  }

  body += renderPhaseLane({
    x: laneX,
    y: y + headerHeight,
    width: laneWidth,
    engine: ENGINE_META.doppler,
    resolved: dopplerPhases,
    globalMax,
  });
  body += renderPhaseLane({
    x: laneX,
    y: y + headerHeight + laneHeight + laneGap,
    width: laneWidth,
    engine: ENGINE_META.transformersjs,
    resolved: transformersPhases,
    globalMax,
  });
  return { body, panelHeight };
}

function renderPhases(rows, width, height, title, subtitle, sectionLabel, sectionPayload) {
  const dopplerPhases = resolvePhaseValues(sectionPayload, ENGINE_META.doppler.key);
  const transformersPhases = resolvePhaseValues(sectionPayload, ENGINE_META.transformersjs.key);
  const globalMax = Math.max(1, dopplerPhases?.endToEnd || 0, transformersPhases?.endToEnd || 0);
  const displaySectionLabel = prettifySectionLabel(sectionLabel);

  let body = '';
  body += renderPhaseSceneDefs();
  body += `<rect x="${CANVAS_PADDING}" y="${CANVAS_PADDING}" width="${width - CANVAS_PADDING * 2}" height="${height - CANVAS_PADDING * 2}" fill="url(#phase-canvas-glow)" stroke="none" />\n`;
  body += `<text x="36" y="54" fill="${ARCHITECTURE_COLORS.edge}" font-family="${FONT_UI}" font-size="12" font-weight="bold" letter-spacing="1.2" stroke="none">PHASE EVIDENCE</text>\n`;
  body += `<text x="36" y="92" fill="${PALETTE.text}" font-family="${FONT_UI}" font-size="30" font-weight="bold" stroke="none">${escapeXml(title)}</text>\n`;
  body += `<text x="36" y="116" fill="${PALETTE.muted}" font-family="${FONT_UI}" font-size="14" stroke="none">${escapeXml(subtitle)}</text>\n`;
  body += `<text x="36" y="136" fill="${PALETTE.muted}" font-family="${FONT_UI}" font-size="12" stroke="none">${escapeXml(`Section: ${displaySectionLabel} • lower is better`)}</text>\n`;
  body += renderPhaseWorkloadPanel({
    x: 28,
    y: 164,
    width: width - 56,
    workloadLabel: subtitle,
    dopplerPhases,
    transformersPhases,
    globalMax,
  }).body;
  return svgWrap(width, height, body, title, `${subtitle} • Section: ${displaySectionLabel}`);
}

function renderMultiPhases(entries, width, title, subtitle) {
  let globalMax = 0;
  const sharedWorkload = entriesShareWorkload(entries);
  const workloads = entries.map((entry) => {
    const dopplerPhases = resolvePhaseValues(entry.sectionPayload, ENGINE_META.doppler.key);
    const transformersPhases = resolvePhaseValues(entry.sectionPayload, ENGINE_META.transformersjs.key);
    globalMax = Math.max(globalMax, dopplerPhases?.endToEnd || 0, transformersPhases?.endToEnd || 0);
    const fallbackWorkload = entry.report.workload?.id || path.basename(entry.inputPath, '.json');
    return {
      workloadLabel: sharedWorkload
        ? (buildModelLabel(entry.report) || buildWorkloadPanelLabel(entry.report, fallbackWorkload))
        : buildWorkloadPanelLabel(entry.report, fallbackWorkload),
      dopplerPhases,
      transformersPhases,
    };
  });

  if (globalMax <= 0) globalMax = 1;

  const panelGap = 22;
  const panelHeight = 190;
  const headerTop = 128;
  const legendHeight = 56;
  const footerHeight = 20;
  const height = headerTop + workloads.length * panelHeight + Math.max(0, workloads.length - 1) * panelGap + legendHeight + footerHeight;

  let body = '';
  body += renderPhaseSceneDefs();
  body += `<rect x="${CANVAS_PADDING}" y="${CANVAS_PADDING}" width="${width - CANVAS_PADDING * 2}" height="${height - CANVAS_PADDING * 2}" fill="url(#phase-canvas-glow)" stroke="none" />\n`;
  body += `<text x="36" y="48" fill="${ARCHITECTURE_COLORS.edge}" font-family="${FONT_UI}" font-size="12" font-weight="bold" letter-spacing="1.2" stroke="none">TIMELINE</text>\n`;
  body += `<text x="36" y="86" fill="${PALETTE.text}" font-family="${FONT_UI}" font-size="30" font-weight="bold" stroke="none">${escapeXml(title)}</text>\n`;
  body += `<text x="36" y="110" fill="${PALETTE.muted}" font-family="${FONT_UI}" font-size="13" stroke="none">${escapeXml(subtitle)}</text>\n`;
  body += `<rect x="${width - 304}" y="36" width="268" height="34" rx="17" fill="url(#phase-winner-pill)" stroke="${PHASE_COLORS.decode}" stroke-width="1.1" />\n`;
  body += `<text x="${width - 170}" y="57" text-anchor="middle" fill="${PALETTE.text}" font-family="${FONT_UI}" font-size="12" font-weight="bold" stroke="none">SHORTER BAR = FASTER</text>\n`;

  workloads.forEach((workload, index) => {
    body += renderPhaseWorkloadPanel({
      x: 24,
      y: headerTop + index * (panelHeight + panelGap),
      width: width - 48,
      workloadLabel: workload.workloadLabel,
      dopplerPhases: workload.dopplerPhases,
      transformersPhases: workload.transformersPhases,
      globalMax,
    }).body;
  });

  const legendY = headerTop + workloads.length * panelHeight + Math.max(0, workloads.length - 1) * panelGap + 26;
  const legendItems = [
    { fill: 'url(#phase-load-grad)', label: 'Load' },
    { fill: 'url(#phase-prefill-grad)', label: 'First token' },
    { fill: 'url(#phase-decode-grad)', label: 'Decode' },
  ];
  legendItems.forEach((item, index) => {
    const x = 36 + index * 184;
    body += `<rect x="${x}" y="${legendY}" width="18" height="18" rx="6" fill="${item.fill}" />\n`;
    body += `<text x="${x + 28}" y="${legendY + 13}" fill="${PALETTE.text}" font-family="${FONT_UI}" font-size="13" stroke="none">${item.label}</text>\n`;
  });
  return svgWrap(width, height, body, title, subtitle);
}

function renderMultiRadar(entries, perRadarHeight, title, subtitle, metricIds) {
  const numWorkloads = entries.length;
  const perRadarWidth = DEFAULT_WIDTH;
  const maxTotalWidth = 1920;
  const maxColumns = Math.max(1, Math.floor(maxTotalWidth / perRadarWidth));
  const columns = Math.max(1, Math.min(numWorkloads, maxColumns));
  const rowsOfCharts = Math.ceil(numWorkloads / columns);
  const totalWidth = perRadarWidth * columns;

  const allScaledData = entries.map((entry) => {
    const allRows = collectRows(entry.report, entry.sectionPayload, metricIds);
    const rows = filterRadarRows(allRows, metricIds);
    const scaledRows = buildScaledRows(rows);
    const fallbackWorkload = entry.report.workload?.id || path.basename(entry.inputPath, '.json');
    const workloadLabel = buildWorkloadPanelLabel(entry.report, fallbackWorkload);
    return { rows, scaledRows, workloadLabel };
  });

  let body = '';
  const headerHeight = 90;
  const legendHeight = 40;
  const chartRowHeight = perRadarHeight + 24;
  const height = headerHeight + rowsOfCharts * chartRowHeight + legendHeight;

  allScaledData.forEach((data, wIndex) => {
    const col = wIndex % columns;
    const row = Math.floor(wIndex / columns);
    const offsetX = col * perRadarWidth;
    const offsetY = headerHeight + row * chartRowHeight;
    const centerX = offsetX + perRadarWidth / 2;
    const centerY = offsetY + (perRadarHeight - 40) / 2;
    const radius = Math.max(40, Math.min(perRadarWidth, perRadarHeight - 40) / 2 - 110);
    const ringCount = 4;
    const axisCount = Math.max(data.rows.length, 1);
    const angleStep = (Math.PI * 2) / axisCount;

    if (data.rows.length < 3) {
      body += `<text x="${centerX}" y="${centerY}" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="12" text-anchor="middle">Need at least 3 metrics.</text>\n`;
      body += `<text x="${centerX}" y="${offsetY + perRadarHeight - 4}" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="12" text-anchor="middle">${escapeXml(data.workloadLabel)}</text>\n`;
      return;
    }

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
      body += `<text x="${x}" y="${y}" fill="${PALETTE.text}" text-anchor="${anchor}" font-family="${FONT_MONO}" font-size="11">${escapeXml(metric.label)}</text>\n`;
    });

    const engines = [ENGINE_META.doppler, ENGINE_META.transformersjs];
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

    body += `<text x="${centerX}" y="${offsetY + perRadarHeight - 4}" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="12" text-anchor="middle">${escapeXml(data.workloadLabel)}</text>\n`;
  });

  const engines = [ENGINE_META.doppler, ENGINE_META.transformersjs];
  const legendBaseY = headerHeight + rowsOfCharts * chartRowHeight + 8;
  const legendX = Math.max(40, totalWidth - 220);
  engines.forEach((engine, i) => {
    const legendY = legendBaseY + i * 16;
    body += `<rect x="${legendX}" y="${legendY - 8}" width="14" height="14" fill="${engine.color}" />\n`;
    body += `<text x="${legendX + 20}" y="${legendY + 2}" fill="${PALETTE.text}" stroke="#ffffff" stroke-width="2" font-family="${FONT_MONO}" font-size="13">${engine.label}</text>\n`;
  });

  body += `<text x="40" y="36" fill="${PALETTE.text}" font-family="${FONT_MONO}" font-size="18" font-weight="bold">${escapeXml(title)}</text>\n`;
  body += `<text x="40" y="54" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="12">${escapeXml(subtitle)}</text>\n`;
  body += `<text x="40" y="70" fill="${PALETTE.muted}" font-family="${FONT_MONO}" font-size="11">All axes are latency (ms), inverted so bigger polygon = faster.</text>\n`;

  return svgWrap(totalWidth, height, body, title, subtitle);
}

const DTYPE_SEGMENT = /^(?:[a-z]*f\d+a?|[a-z]*q\d+[a-z]*|bf16|fp16|fp32|int[48])$/i;

function prettifyModelId(raw, { stripDtype = false } = {}) {
  if (!raw) return 'unknown';
  const stripped = raw
    .replace(/^[^/]+\//, '')
    .replace(/-ONNX-GQA$/i, '')
    .replace(/-ONNX$/i, '');
  const lowerParts = stripped.toLowerCase().split('-');
  const parts = stripDtype
    ? lowerParts.filter((part) => !DTYPE_SEGMENT.test(part))
    : lowerParts;
  const mergedParts = [];

  for (let i = 0; i < parts.length; i += 1) {
    const part = parts[i];
    const next = parts[i + 1];

    if (/^lfm\d+$/.test(part) && next && /^\d+$/.test(next)) {
      mergedParts.push(`${part}.${next}`);
      i += 1;
      continue;
    }
    const prev = i > 0 ? parts[i - 1] : null;
    if (/^\d+$/.test(part) && next && /^\d+b$/.test(next) && prev !== 'gemma') {
      mergedParts.push(`${part}.${next}`);
      i += 1;
      continue;
    }
    mergedParts.push(part);
  }

  if (stripDtype) {
    while (mergedParts.length > 1 && DTYPE_SEGMENT.test(mergedParts[mergedParts.length - 1])) {
      mergedParts.pop();
    }
  }
  return mergedParts
    .map((part) => {
      if (part === 'gemma') return 'Gemma';
      if (part === 'it' || part === 'instruct') return null;
      if (/^lfm\d+(?:\.\d+)?$/.test(part)) {
        return `LFM ${part.slice(3)}`;
      }
      return part;
    })
    .filter((part) => typeof part === 'string' && part.length > 0)
    .join(' ');
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
  return parts.length > 0 ? parts.join(', ') : workload.id || EMPTY_STRING;
}

function prettifyCompactWorkload(workload) {
  if (!workload) return '';
  const parts = [];
  const prefill = workload.prefillTokenTarget ?? workload.prefillTokens;
  const decode = workload.decodeTokenTarget ?? workload.decodeTokens;
  if (isFiniteNumber(prefill) && isFiniteNumber(decode)) {
    parts.push(`${prefill} prompt / ${decode} decode`);
  } else {
    const verbose = prettifyWorkload(workload);
    if (verbose.length > 0) return verbose;
  }
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
  return parts.length > 0 ? parts.join(' • ') : workload.id || EMPTY_STRING;
}

function formatDtypeSuffix(dtype) {
  if (!dtype) return '';
  const lower = dtype.toLowerCase();
  const normalized = lower === 'f32' ? 'f32a' : lower;
  return `, ${normalized}`;
}

function normalizeTjsDtype(raw) {
  if (!raw) return null;
  const lower = raw.toLowerCase();
  if (lower.includes('/')) return lower;
  return `${lower}/f32a`;
}

function buildTitle() {
  return STATIC_CHART_TITLE;
}

function buildModelLabel(report) {
  const dopplerModelId = report.dopplerModelId || report.modelId || EMPTY_STRING;
  const dopplerName = prettifyModelId(dopplerModelId, { stripDtype: true });
  return dopplerName;
}

function buildWorkloadPanelLabel(report, fallbackWorkload) {
  const model = buildModelLabel(report);
  const workload = prettifyWorkload(report.workload) || fallbackWorkload || EMPTY_STRING;
  return workload.length > 0 ? `${model} • ${workload}` : model;
}

function buildHeaderLabel(report) {
  const model = buildModelLabel(report);
  const workloadDesc = prettifyWorkload(report.workload);
  const workloadPart = workloadDesc ? ` • ${workloadDesc}` : '';
  return `${model}${workloadPart}`;
}

function buildSubtitle(report, inputPath) {
  return buildHeaderLabel(report);
}

function prettifySectionLabel(sectionLabel) {
  const normalized = String(sectionLabel || EMPTY_STRING).trim();
  if (normalized === 'compute/parity') return 'warm parity';
  if (normalized === 'compute/throughput') return 'warm throughput';
  return normalized;
}

function buildMultiPhaseSubtitle(entries) {
  const sharedWorkload = entriesShareWorkload(entries);
  const workload = sharedWorkload ? prettifyCompactWorkload(entries[0]?.report?.workload) : '';
  const models = [...new Set(entries.map((entry) => buildModelLabel(entry.report)).filter((value) => value.length > 0))];
  if (models.length === 0) return workload ? `Warm cache • ${workload}` : 'Warm cache';
  return workload ? `${models.join(' + ')} • warm cache • ${workload}` : models.join(' + ');
}

function entriesShareWorkload(entries) {
  const workloads = [...new Set(entries.map((entry) => prettifyWorkload(entry.report?.workload)).filter((value) => value.length > 0))];
  return workloads.length <= 1;
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
  applyScenarioOptions(options);
  if (options.inputs.length === 0) {
    throw new Error('Missing required --input path.');
  }
  const excludeSet = new Set(options.excludeWorkloads.map((item) => normalizeWorkloadFilterToken(item)));
  const includeSet = new Set(options.includeWorkloads.map((item) => normalizeWorkloadFilterToken(item)));

  const rawEntries = options.inputs.map((raw) => {
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
  const entries = rawEntries
    .filter((entry) => shouldIncludeWorkload(entry, includeSet))
    .filter((entry) => !shouldExcludeWorkload(entry, excludeSet));

  if (entries.length === 0) {
    throw new Error('No workload data remains after applying workload filters.');
  }
  assertComparableEntries(entries, { allowNonComparable: options.allowNonComparable });

  const firstEntry = entries[0];
  const title = buildTitle();
  const headerLabel = buildHeaderLabel(firstEntry.report);
  const isMulti = entries.length > 1;

  let svg;
  if ((options.chart === 'phases' || options.chart === 'radar') && isMulti) {
    const subtitle = options.chart === 'phases'
      ? buildMultiPhaseSubtitle(entries)
      : headerLabel;

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

export {
  loadChartMetricContract,
  resolveSection,
  parseArgs,
  main,
};
