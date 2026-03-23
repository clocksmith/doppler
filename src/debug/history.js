import {
  LOG_LEVELS,
  currentLogLevel,
  enabledTraceCategories,
  enabledModules,
  disabledModules,
  logHistory,
  getLogHistoryLimit,
} from './config.js';

// ============================================================================
// History Functions
// ============================================================================

/**
 * Enforce the max history size by evicting oldest entries first.
 * This is a safety net in case entries were pushed from multiple
 * concurrent paths between limit checks in storeLog/storeTrace.
 */
function enforceHistoryLimit() {
  const maxHistory = getLogHistoryLimit();
  while (logHistory.length > maxHistory) {
    logHistory.shift();
  }
}

export function getLogHistory(filter = {}) {
  enforceHistoryLimit();
  let history = [...logHistory];

  if (filter.level) {
    history = history.filter((h) => h.level === filter.level.toUpperCase());
  }

  if (filter.module) {
    const m = filter.module.toLowerCase();
    history = history.filter((h) => h.module.toLowerCase().includes(m));
  }

  if (filter.last) {
    history = history.slice(-filter.last);
  }

  return history;
}


export function clearLogHistory() {
  logHistory.length = 0;
}


export function printLogSummary(count = 20) {
  const recent = logHistory.slice(-count);
  console.log('=== Recent Logs ===');
  for (const entry of recent) {
    const time = entry.perfTime.toFixed(1);
    console.log(`[${time}ms][${entry.level}][${entry.module}] ${entry.message}`);
  }
  console.log('===================');
}


export function getDebugSnapshot() {
  return {
    timestamp: new Date().toISOString(),
    logLevel: Object.keys(LOG_LEVELS).find(
      (k) => LOG_LEVELS[k] === currentLogLevel
    ),
    traceCategories: [...enabledTraceCategories],
    enabledModules: [...enabledModules],
    disabledModules: [...disabledModules],
    recentLogs: logHistory.slice(-50).map((e) => ({
      time: e.perfTime.toFixed(1),
      level: e.level,
      module: e.module,
      message: e.message,
    })),
    errorCount: logHistory.filter((e) => e.level === 'ERROR').length,
    warnCount: logHistory.filter((e) => e.level === 'WARN').length,
  };
}
