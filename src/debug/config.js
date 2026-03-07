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
  'energy',
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

function requirePlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object when provided.`);
  }
  return value;
}

function requireNonNegativeIntegerArray(value, label) {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array of non-negative integers when provided.`);
  }
  return value.map((entry, index) => {
    const parsed = Number(entry);
    if (!Number.isInteger(parsed) || parsed < 0) {
      throw new Error(`${label}[${index}] must be a non-negative integer.`);
    }
    return parsed;
  });
}

function requireNonNegativeInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative integer when provided.`);
  }
  return parsed;
}

function requireBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new Error(`${label} must be a boolean when provided.`);
  }
  return value;
}

function normalizeLogLevel(level) {
  if (typeof level !== 'string' || !level.trim()) {
    throw new Error('setLogLevel(level) requires a non-empty log level string.');
  }
  return level.trim().toLowerCase();
}

function normalizeTraceCategories(categories) {
  if (typeof categories === 'string') {
    const values = categories
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean);
    if (values.length === 0) {
      throw new Error('setTrace(categories) requires at least one trace category.');
    }
    return values;
  }
  if (Array.isArray(categories) && categories.length > 0) {
    return categories.map((value, index) => {
      if (typeof value !== 'string' || !value.trim()) {
        throw new Error(`setTrace(categories)[${index}] must be a non-empty string.`);
      }
      return value.trim();
    });
  }
  throw new Error(
    'setTrace(categories) requires false, a comma-delimited string, or a non-empty string array.'
  );
}

function validateTraceCategoryToken(token) {
  if (token === 'all') {
    return;
  }
  const value = token.startsWith('-') ? token.slice(1) : token;
  if (!TRACE_CATEGORIES.includes(value)) {
    throw new Error(
      `Unknown trace category "${token}". Allowed categories: all, ${TRACE_CATEGORIES.join(', ')}.`
    );
  }
}

function normalizeTraceOptions(options) {
  if (options == null) {
    return {};
  }
  const normalized = requirePlainObject(options, 'setTrace(options)');
  return {
    ...(normalized.layers === undefined ? {} : {
      layers: requireNonNegativeIntegerArray(normalized.layers, 'setTrace(options).layers'),
    }),
    ...(normalized.maxDecodeSteps === undefined ? {} : {
      maxDecodeSteps: requireNonNegativeInteger(
        normalized.maxDecodeSteps,
        'setTrace(options).maxDecodeSteps'
      ),
    }),
    ...(normalized.breakOnAnomaly === undefined ? {} : {
      breakOnAnomaly: requireBoolean(
        normalized.breakOnAnomaly,
        'setTrace(options).breakOnAnomaly'
      ),
    }),
  };
}

export function setLogLevel(level) {
  const levelMap = {
    debug: LOG_LEVELS.DEBUG,
    verbose: LOG_LEVELS.VERBOSE,
    info: LOG_LEVELS.INFO,
    warn: LOG_LEVELS.WARN,
    error: LOG_LEVELS.ERROR,
    silent: LOG_LEVELS.SILENT,
  };
  const normalizedLevel = normalizeLogLevel(level);
  if (!Object.prototype.hasOwnProperty.call(levelMap, normalizedLevel)) {
    throw new Error(
      `Unknown log level "${level}". Allowed levels: ${Object.keys(levelMap).join(', ')}.`
    );
  }
  currentLogLevel = levelMap[normalizedLevel];
  console.log(`[Doppler] Log level set to: ${normalizedLevel.toUpperCase()}`);
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

  const catArray = normalizeTraceCategories(categories);
  const traceOptions = normalizeTraceOptions(options);
  for (const cat of catArray) {
    validateTraceCategoryToken(cat);
  }

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

  if (traceOptions.layers !== undefined) {
    traceLayerFilter = traceOptions.layers;
  }
  if (traceOptions.maxDecodeSteps !== undefined) {
    traceMaxDecodeSteps = traceOptions.maxDecodeSteps;
  }
  if (traceOptions.breakOnAnomaly !== undefined) {
    traceBreakOnAnomaly = traceOptions.breakOnAnomaly;
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
    console.warn = noop;
    originalConsoleLog('[Doppler] Silent mode enabled - logging silenced');
  } else {
    console.log = originalConsoleLog;
    console.debug = originalConsoleDebug;
    console.info = originalConsoleInfo;
    console.warn = originalConsoleWarn;
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
