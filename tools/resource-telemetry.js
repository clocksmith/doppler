import fs from 'node:fs';
import os from 'node:os';
import { spawnSync } from 'node:child_process';

const DEFAULT_RESOURCE_TELEMETRY_INTERVAL_MS = 500;
const DEFAULT_CLOCK_TICKS_PER_SECOND = 100;
const PROC_ROOT = '/proc';

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function asFiniteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function roundNumber(value, digits = 3) {
  if (!Number.isFinite(value)) return null;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function parsePositiveInteger(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseResourceTelemetryMode(value, fallback = false, label = '--resource-telemetry') {
  if (value == null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'on' || normalized === 'true' || normalized === '1' || normalized === 'yes') return true;
  if (normalized === 'off' || normalized === 'false' || normalized === '0' || normalized === 'no') return false;
  throw new Error(`${label} must be one of: on, off, true, false, 1, 0`);
}

function commandExists(command) {
  const result = spawnSync('sh', ['-lc', `command -v ${command}`], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
    timeout: 2000,
  });
  return result.status === 0 && String(result.stdout || '').trim() !== '';
}

function resolveClockTicksPerSecond() {
  const result = spawnSync('getconf', ['CLK_TCK'], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
    timeout: 2000,
  });
  const value = Number(String(result.stdout || '').trim());
  return Number.isFinite(value) && value > 0 ? value : DEFAULT_CLOCK_TICKS_PER_SECOND;
}

function parseKbField(text, key) {
  const match = new RegExp(`^${key}:\\s+(\\d+)\\s+kB`, 'm').exec(text);
  return match ? Number(match[1]) * 1024 : null;
}

function parseProcStat(text, clockTicksPerSecond) {
  const closeParen = text.lastIndexOf(')');
  if (closeParen < 0) return null;
  const before = text.slice(0, closeParen + 1);
  const after = text.slice(closeParen + 1).trim().split(/\s+/);
  const pid = Number(before.split(/\s+/, 1)[0]);
  const ppid = Number(after[1]);
  const userTicks = Number(after[11]);
  const systemTicks = Number(after[12]);
  if (!Number.isInteger(pid) || !Number.isInteger(ppid)) return null;
  const userMs = Number.isFinite(userTicks) ? (userTicks / clockTicksPerSecond) * 1000 : null;
  const systemMs = Number.isFinite(systemTicks) ? (systemTicks / clockTicksPerSecond) * 1000 : null;
  return {
    pid,
    ppid,
    userMs,
    systemMs,
    totalCpuMs: Number.isFinite(userMs) && Number.isFinite(systemMs) ? userMs + systemMs : null,
  };
}

function readProcProcessRows(clockTicksPerSecond) {
  if (process.platform !== 'linux' || !fs.existsSync(PROC_ROOT)) {
    return { rows: [], unavailableReason: 'procfs-unavailable' };
  }
  let entries;
  try {
    entries = fs.readdirSync(PROC_ROOT, { withFileTypes: true });
  } catch {
    return { rows: [], unavailableReason: 'procfs-readdir-failed' };
  }

  const rows = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || !/^\d+$/.test(entry.name)) continue;
    const pid = Number(entry.name);
    const basePath = `${PROC_ROOT}/${entry.name}`;
    let stat;
    try {
      stat = parseProcStat(fs.readFileSync(`${basePath}/stat`, 'utf8'), clockTicksPerSecond);
    } catch {
      continue;
    }
    if (!stat || stat.pid !== pid) continue;
    let status = '';
    try {
      status = fs.readFileSync(`${basePath}/status`, 'utf8');
    } catch {
      status = '';
    }
    rows.push({
      pid,
      ppid: stat.ppid,
      rssBytes: parseKbField(status, 'VmRSS'),
      hwmBytes: parseKbField(status, 'VmHWM'),
      userMs: stat.userMs,
      systemMs: stat.systemMs,
      totalCpuMs: stat.totalCpuMs,
    });
  }
  return { rows, unavailableReason: null };
}

function collectProcessTree(rows, rootPid) {
  const childrenByParent = new Map();
  const rowByPid = new Map();
  for (const row of rows) {
    rowByPid.set(row.pid, row);
    if (!childrenByParent.has(row.ppid)) childrenByParent.set(row.ppid, []);
    childrenByParent.get(row.ppid).push(row.pid);
  }
  const seen = new Set();
  const stack = [rootPid];
  while (stack.length > 0) {
    const pid = stack.pop();
    if (seen.has(pid)) continue;
    seen.add(pid);
    const children = childrenByParent.get(pid) || [];
    for (const childPid of children) {
      stack.push(childPid);
    }
  }
  const treeRows = [...seen].map((pid) => rowByPid.get(pid)).filter(Boolean);
  const sumFinite = (field) => {
    let sum = 0;
    let count = 0;
    for (const row of treeRows) {
      if (Number.isFinite(row[field])) {
        sum += row[field];
        count += 1;
      }
    }
    return count > 0 ? sum : null;
  };
  const maxFinite = (field) => {
    const values = treeRows.map((row) => row[field]).filter(Number.isFinite);
    return values.length > 0 ? Math.max(...values) : null;
  };
  return {
    pid: rootPid,
    processCount: treeRows.length,
    rssBytes: sumFinite('rssBytes'),
    hwmBytes: maxFinite('hwmBytes'),
    userMs: sumFinite('userMs'),
    systemMs: sumFinite('systemMs'),
    totalCpuMs: sumFinite('totalCpuMs'),
  };
}

function readMeminfo() {
  if (process.platform !== 'linux' || !fs.existsSync(`${PROC_ROOT}/meminfo`)) {
    return { value: null, unavailableReason: 'meminfo-unavailable' };
  }
  let text;
  try {
    text = fs.readFileSync(`${PROC_ROOT}/meminfo`, 'utf8');
  } catch {
    return { value: null, unavailableReason: 'meminfo-read-failed' };
  }
  const totalBytes = parseKbField(text, 'MemTotal');
  const availableBytes = parseKbField(text, 'MemAvailable');
  const freeBytes = parseKbField(text, 'MemFree');
  const usedBytes = Number.isFinite(totalBytes) && Number.isFinite(availableBytes)
    ? totalBytes - availableBytes
    : null;
  return {
    value: {
      ramTotalBytes: totalBytes,
      ramAvailableBytes: availableBytes,
      ramFreeBytes: freeBytes,
      ramUsedBytes: usedBytes,
    },
    unavailableReason: null,
  };
}

function parseRocmNumber(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string') return null;
  const match = /[-+]?\d+(?:\.\d+)?/.exec(value);
  return match ? Number(match[0]) : null;
}

function findRocmValue(row, patterns) {
  if (!isPlainObject(row)) return null;
  for (const [key, value] of Object.entries(row)) {
    const normalized = key.toLowerCase();
    if (patterns.some((pattern) => pattern.test(normalized))) {
      return parseRocmNumber(value);
    }
  }
  return null;
}

function parseRocmSmiJson(text) {
  let parsed;
  try {
    parsed = JSON.parse(String(text || '').trim());
  } catch {
    return null;
  }
  if (!isPlainObject(parsed)) return null;
  const devices = [];
  for (const [id, row] of Object.entries(parsed)) {
    if (!isPlainObject(row)) continue;
    const utilizationPercent = findRocmValue(row, [/gpu use \(%\)/]);
    const memoryUsedBytes = findRocmValue(row, [/vram total used memory \(b\)/]);
    const memoryTotalBytes = findRocmValue(row, [/vram total memory \(b\)/]);
    const memoryUsedPercent = findRocmValue(row, [/gpu memory allocated.*vram%\)/]);
    const powerWatts = findRocmValue(row, [/power \(w\)/]);
    devices.push({
      id,
      utilizationPercent,
      memoryUsedBytes,
      memoryTotalBytes,
      memoryUsedPercent,
      powerWatts,
    });
  }
  if (devices.length === 0) return null;
  const sum = (field) => {
    const values = devices.map((device) => device[field]).filter(Number.isFinite);
    return values.length > 0 ? values.reduce((acc, value) => acc + value, 0) : null;
  };
  const mean = (field) => {
    const values = devices.map((device) => device[field]).filter(Number.isFinite);
    return values.length > 0 ? values.reduce((acc, value) => acc + value, 0) / values.length : null;
  };
  return {
    provider: 'rocm-smi',
    deviceCount: devices.length,
    devices,
    utilizationPercent: mean('utilizationPercent'),
    memoryUsedBytes: sum('memoryUsedBytes'),
    memoryTotalBytes: sum('memoryTotalBytes'),
    memoryUsedPercent: mean('memoryUsedPercent'),
    powerWatts: sum('powerWatts'),
  };
}

function readRocmSmiSnapshot(rocmSmiAvailable) {
  if (!rocmSmiAvailable) {
    return { value: null, unavailableReason: 'rocm-smi-unavailable' };
  }
  const result = spawnSync('rocm-smi', [
    '--showuse',
    '--showmemuse',
    '--showmeminfo',
    'vram',
    '--showpower',
    '--json',
  ], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
    timeout: 5000,
    maxBuffer: 1024 * 1024,
  });
  if (result.status !== 0) {
    return { value: null, unavailableReason: 'rocm-smi-failed' };
  }
  const value = parseRocmSmiJson(result.stdout);
  return value
    ? { value, unavailableReason: null }
    : { value: null, unavailableReason: 'rocm-smi-json-unavailable' };
}

function summarizeSeries(values, digits = 3) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return null;
  const sum = finite.reduce((acc, value) => acc + value, 0);
  return {
    min: roundNumber(Math.min(...finite), digits),
    max: roundNumber(Math.max(...finite), digits),
    mean: roundNumber(sum / finite.length, digits),
    last: roundNumber(finite[finite.length - 1], digits),
  };
}

function summarizeIntegerSeries(values) {
  const summary = summarizeSeries(values, 0);
  if (!summary) return null;
  return summary;
}

function collectUnavailableReasons(samples, setupReasons = []) {
  const reasons = new Set(setupReasons.filter(Boolean));
  for (const sample of samples) {
    for (const reason of sample.unavailableReasons || []) {
      if (reason) reasons.add(reason);
    }
  }
  return [...reasons].sort();
}

function summarizeResourceTelemetrySamples({
  samples,
  pid,
  label,
  intervalMs,
  includeSamples,
  startedAtMs,
  endedAtMs,
  sources,
  setupUnavailableReasons = [],
}) {
  const unavailableReasons = collectUnavailableReasons(samples, setupUnavailableReasons);
  const durationMs = Number.isFinite(startedAtMs) && Number.isFinite(endedAtMs)
    ? Math.max(0, endedAtMs - startedAtMs)
    : null;
  const processSamples = samples.map((sample) => sample.process).filter(isPlainObject);
  const systemSamples = samples.map((sample) => sample.system).filter(isPlainObject);
  const gpuSamples = samples.map((sample) => sample.gpu).filter(isPlainObject);
  const summary = {
    schemaVersion: 1,
    enabled: true,
    label: label ?? null,
    startedAt: Number.isFinite(startedAtMs) ? new Date(startedAtMs).toISOString() : null,
    endedAt: Number.isFinite(endedAtMs) ? new Date(endedAtMs).toISOString() : null,
    durationMs: durationMs == null ? null : Math.round(durationMs),
    sampling: {
      intervalMs,
      sampleCount: samples.length,
      includeSamples: includeSamples === true,
      sources,
      unavailableReasons,
    },
    process: {
      pid,
      processCount: summarizeIntegerSeries(processSamples.map((sample) => sample.processCount)),
      rssBytes: summarizeIntegerSeries(processSamples.map((sample) => sample.rssBytes)),
      hwmBytes: summarizeIntegerSeries(processSamples.map((sample) => sample.hwmBytes)),
      cpuUserMs: summarizeSeries(processSamples.map((sample) => sample.userMs)),
      cpuSystemMs: summarizeSeries(processSamples.map((sample) => sample.systemMs)),
      cpuTotalMs: summarizeSeries(processSamples.map((sample) => sample.totalCpuMs)),
      cpuPercent: summarizeSeries(processSamples.map((sample) => sample.cpuPercent)),
    },
    system: {
      ramTotalBytes: (() => {
        const values = systemSamples.map((sample) => sample.ramTotalBytes).filter(Number.isFinite);
        return values.length > 0 ? values[values.length - 1] : null;
      })(),
      ramAvailableBytes: summarizeIntegerSeries(systemSamples.map((sample) => sample.ramAvailableBytes)),
      ramFreeBytes: summarizeIntegerSeries(systemSamples.map((sample) => sample.ramFreeBytes)),
      ramUsedBytes: summarizeIntegerSeries(systemSamples.map((sample) => sample.ramUsedBytes)),
    },
    gpu: gpuSamples.length > 0
      ? {
          provider: 'rocm-smi',
          deviceCount: gpuSamples[gpuSamples.length - 1].deviceCount ?? null,
          utilizationPercent: summarizeSeries(gpuSamples.map((sample) => sample.utilizationPercent)),
          memoryUsedBytes: summarizeIntegerSeries(gpuSamples.map((sample) => sample.memoryUsedBytes)),
          memoryTotalBytes: (() => {
            const values = gpuSamples.map((sample) => sample.memoryTotalBytes).filter(Number.isFinite);
            return values.length > 0 ? values[values.length - 1] : null;
          })(),
          memoryUsedPercent: summarizeSeries(gpuSamples.map((sample) => sample.memoryUsedPercent)),
          powerWatts: summarizeSeries(gpuSamples.map((sample) => sample.powerWatts)),
        }
      : null,
  };
  if (includeSamples === true) {
    summary.samples = samples;
  }
  return summary;
}

function normalizeResourceTelemetryOptions(options = {}) {
  const enabled = options.enabled === true;
  return {
    enabled,
    intervalMs: parsePositiveInteger(
      options.intervalMs,
      DEFAULT_RESOURCE_TELEMETRY_INTERVAL_MS,
      'resource telemetry interval'
    ),
    includeSamples: options.includeSamples === true,
    includeGpu: options.includeGpu !== false,
    label: typeof options.label === 'string' && options.label.trim() !== '' ? options.label.trim() : null,
  };
}

function createResourceTelemetrySampler(options = {}) {
  const normalized = normalizeResourceTelemetryOptions(options);
  if (!normalized.enabled) {
    return {
      start() {},
      stop() {
        return null;
      },
    };
  }

  const clockTicksPerSecond = resolveClockTicksPerSecond();
  const rocmSmiAvailable = normalized.includeGpu === true && commandExists('rocm-smi');
  const setupUnavailableReasons = [];
  if (process.platform !== 'linux') setupUnavailableReasons.push('linux-procfs-required');
  if (normalized.includeGpu === true && !rocmSmiAvailable) setupUnavailableReasons.push('rocm-smi-unavailable');
  const sources = {
    procfs: process.platform === 'linux' && fs.existsSync(PROC_ROOT),
    meminfo: process.platform === 'linux' && fs.existsSync(`${PROC_ROOT}/meminfo`),
    rocmSmi: rocmSmiAvailable,
  };

  let pid = null;
  let timer = null;
  let startedAtMs = null;
  let previousSample = null;
  const samples = [];

  const sample = () => {
    if (!Number.isInteger(pid) || pid <= 0) return;
    const timestampMs = Date.now();
    const unavailableReasons = [];
    const procRows = readProcProcessRows(clockTicksPerSecond);
    if (procRows.unavailableReason) unavailableReasons.push(procRows.unavailableReason);
    const processTree = collectProcessTree(procRows.rows, pid);
    const elapsedMs = previousSample ? timestampMs - previousSample.timestampMs : null;
    const cpuDeltaMs = previousSample?.process
      && Number.isFinite(processTree.totalCpuMs)
      && Number.isFinite(previousSample.process.totalCpuMs)
      ? processTree.totalCpuMs - previousSample.process.totalCpuMs
      : null;
    const cpuPercent = Number.isFinite(elapsedMs) && elapsedMs > 0 && Number.isFinite(cpuDeltaMs) && cpuDeltaMs >= 0
      ? (cpuDeltaMs / elapsedMs) * 100
      : null;
    const meminfo = readMeminfo();
    if (meminfo.unavailableReason) unavailableReasons.push(meminfo.unavailableReason);
    const gpuSnapshot = readRocmSmiSnapshot(rocmSmiAvailable);
    if (gpuSnapshot.unavailableReason) unavailableReasons.push(gpuSnapshot.unavailableReason);
    const currentSample = {
      timestamp: new Date(timestampMs).toISOString(),
      timestampMs,
      process: {
        ...processTree,
        cpuPercent,
      },
      system: meminfo.value,
      gpu: gpuSnapshot.value,
      unavailableReasons,
    };
    samples.push(currentSample);
    previousSample = currentSample;
  };

  return {
    start(rootPid) {
      if (!Number.isInteger(rootPid) || rootPid <= 0) {
        setupUnavailableReasons.push('invalid-root-pid');
        return;
      }
      pid = rootPid;
      startedAtMs = Date.now();
      sample();
      timer = setInterval(sample, normalized.intervalMs);
      if (typeof timer.unref === 'function') timer.unref();
    },
    stop() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
      sample();
      return summarizeResourceTelemetrySamples({
        samples,
        pid,
        label: normalized.label,
        intervalMs: normalized.intervalMs,
        includeSamples: normalized.includeSamples,
        startedAtMs,
        endedAtMs: Date.now(),
        sources,
        setupUnavailableReasons,
      });
    },
  };
}

export {
  DEFAULT_RESOURCE_TELEMETRY_INTERVAL_MS,
  createResourceTelemetrySampler,
  normalizeResourceTelemetryOptions,
  parseResourceTelemetryMode,
  parseRocmSmiJson,
  summarizeResourceTelemetrySamples,
};
