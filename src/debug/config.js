import { DEFAULT_LOG_HISTORY_CONFIG } from '../config/schema/debug.schema.js';

// Log level values (higher = less verbose)
export const LOG_LEVELS = {
  DEBUG: 0,
  VERBOSE: 1,
  INFO: 2,
  WARN: 3,
  ERROR: 4,
  SILENT: 5,
};

// Trace categories
export const TRACE_CATEGORIES = [
  'loader',
  'kernels',
  'logits',
  'embed',
  'attn',
  'ffn',
  'kv',
  'sample',
  'buffers',
  'perf',
];

// Global state
export let currentLogLevel = LOG_LEVELS.INFO;
let logHistoryLimit = DEFAULT_LOG_HISTORY_CONFIG.maxLogHistoryEntries;
export let enabledModules = new Set();
export let disabledModules = new Set();
export let logHistory = [];

// GPU device reference for tensor inspection
export let gpuDevice = null;

// Trace categories state
export let enabledTraceCategories = new Set();
export let traceLayerFilter = [];
export let traceDecodeStep = 0;
export let traceMaxDecodeSteps = 0;
export let traceBreakOnAnomaly = false;

// Benchmark mode state (silent mode)
let silentMode = false;
const originalConsoleLog = console.log;
const originalConsoleDebug = console.debug;
const originalConsoleInfo = console.info;
const originalConsoleWarn = console.warn;
let warnedBenchmarkMode = false;

export function setLogLevel(level) {
  const levelMap = {
    debug: LOG_LEVELS.DEBUG,
    verbose: LOG_LEVELS.VERBOSE,
    info: LOG_LEVELS.INFO,
    warn: LOG_LEVELS.WARN,
    error: LOG_LEVELS.ERROR,
    silent: LOG_LEVELS.SILENT,
  };
  currentLogLevel = levelMap[level.toLowerCase()] ?? LOG_LEVELS.INFO;
  console.log(`[Doppler] Log level set to: ${level.toUpperCase()}`);
}

export function getLogLevel() {
  for (const [name, value] of Object.entries(LOG_LEVELS)) {
    if (value === currentLogLevel) return name.toLowerCase();
  }
  return 'info';
}

export function setTrace(categories, options) {
  if (categories === false) {
    enabledTraceCategories.clear();
    console.log('[Doppler] Trace disabled');
    return;
  }

  const catArray = typeof categories === 'string'
    ? categories.split(',').map(s => s.trim())
    : categories;

  enabledTraceCategories.clear();

  const hasAll = catArray.includes('all');
  if (hasAll) {
    for (const cat of TRACE_CATEGORIES) {
      enabledTraceCategories.add(cat);
    }
  }

  for (const cat of catArray) {
    if (cat === 'all') continue;

    if (cat.startsWith('-')) {
      const exclude = cat.slice(1);
      enabledTraceCategories.delete(exclude);
    } else if (TRACE_CATEGORIES.includes(cat)) {
      enabledTraceCategories.add(cat);
    }
  }

  if (options?.layers) {
    traceLayerFilter = options.layers;
  }
  if (options?.maxDecodeSteps !== undefined) {
    traceMaxDecodeSteps = options.maxDecodeSteps;
  }
  if (options?.breakOnAnomaly !== undefined) {
    traceBreakOnAnomaly = options.breakOnAnomaly;
  }

  const enabled = [...enabledTraceCategories].join(',') || 'none';
  console.log(`[Doppler] Trace categories: ${enabled}`);
}

export function applyDebugConfig(config) {
  const logHistoryConfig = config?.logHistory ?? DEFAULT_LOG_HISTORY_CONFIG;
  logHistoryLimit = logHistoryConfig.maxLogHistoryEntries ?? DEFAULT_LOG_HISTORY_CONFIG.maxLogHistoryEntries;

  if (config.logLevel?.defaultLogLevel) {
    const desired = config.logLevel.defaultLogLevel;
    if (desired && desired !== getLogLevel()) {
      setLogLevel(desired);
    }
  }

  if (config.trace?.enabled) {
    const categories = config.trace.categories?.length
      ? config.trace.categories.join(',')
      : 'all';
    setTrace(categories, {
      layers: config.trace.layers ?? undefined,
      maxDecodeSteps: config.trace.maxDecodeSteps || undefined,
    });
  } else if (getTrace().length > 0) {
    setTrace(false);
  }
}

export function getLogHistoryLimit() {
  return logHistoryLimit;
}

export function getTrace() {
  return [...enabledTraceCategories];
}

export function isTraceEnabled(category, layerIdx) {
  if (!enabledTraceCategories.has(category)) return false;

  if (layerIdx !== undefined && traceLayerFilter.length > 0) {
    if (!traceLayerFilter.includes(layerIdx)) return false;
  }

  if (traceMaxDecodeSteps > 0 && traceDecodeStep > traceMaxDecodeSteps) {
    return false;
  }

  return true;
}

export function incrementDecodeStep() {
  return ++traceDecodeStep;
}

export function resetDecodeStep() {
  traceDecodeStep = 0;
}

export function getDecodeStep() {
  return traceDecodeStep;
}

export function shouldBreakOnAnomaly() {
  return traceBreakOnAnomaly;
}

export function setSilentMode(enabled) {
  silentMode = enabled;
  if (enabled) {
    const noop = () => {};
    console.log = noop;
    console.debug = noop;
    console.info = noop;
    originalConsoleLog('[Doppler] Silent mode enabled - logging silenced');
  } else {
    console.log = originalConsoleLog;
    console.debug = originalConsoleDebug;
    console.info = originalConsoleInfo;
    console.log('[Doppler] Silent mode disabled - logging restored');
  }
}

export function isSilentMode() {
  return silentMode;
}

export function setBenchmarkMode(enabled) {
  if (!warnedBenchmarkMode) {
    warnedBenchmarkMode = true;
    originalConsoleWarn('[Doppler] setBenchmarkMode is deprecated; use setSilentMode instead.');
  }
  setSilentMode(enabled);
}

export function isBenchmarkMode() {
  if (!warnedBenchmarkMode) {
    warnedBenchmarkMode = true;
    originalConsoleWarn('[Doppler] isBenchmarkMode is deprecated; use isSilentMode instead.');
  }
  return isSilentMode();
}

export function enableModules(...modules) {
  enabledModules = new Set(modules.map((m) => m.toLowerCase()));
  console.log(`[Doppler] Enabled modules: ${modules.join(', ')}`);
}

export function disableModules(...modules) {
  for (const m of modules) {
    disabledModules.add(m.toLowerCase());
  }
  console.log(`[Doppler] Disabled modules: ${modules.join(', ')}`);
}

export function resetModuleFilters() {
  enabledModules.clear();
  disabledModules.clear();
}

export function setGPUDevice(device) {
  gpuDevice = device;
}
