

import {
  LOG_LEVELS,
  currentLogLevel,
  enabledModules,
  disabledModules,
  logHistory,
  getLogHistoryLimit,
} from './config.js';

let diagnosticObserver = null;

export function resolveDiagnosticObserver(observer) {
  if (observer == null) return null;
  if (typeof observer.observe !== 'function') throw new Error('Diagnostic observer requires observe(event).');
  return Object.freeze({ observe: observer.observe.bind(observer) });
}

export function enterDiagnosticObserver(observer) {
  const previous = diagnosticObserver;
  diagnosticObserver = observer;
  return () => { diagnosticObserver = previous; };
}

export function emitDiagnostic(event) {
  if (!diagnosticObserver) return false;
  diagnosticObserver.observe(Object.freeze({ type: 'diagnostic', ...event }));
  return true;
}

// ============================================================================
// Internal Helpers
// ============================================================================


function shouldLog(module, level) {
  if (level < currentLogLevel) return false;

  const moduleLower = module.toLowerCase();

  if (enabledModules.size > 0 && !enabledModules.has(moduleLower)) {
    return false;
  }

  if (disabledModules.has(moduleLower)) {
    return false;
  }

  return true;
}


function formatMessage(module, message) {
  const timestamp = performance.now().toFixed(1);
  return `[${timestamp}ms][${module}] ${message}`;
}


function storeLog(level, module, message, data) {
  logHistory.push({
    time: Date.now(),
    perfTime: performance.now(),
    level,
    module,
    message,
    data,
  });

  const maxHistory = getLogHistoryLimit();
  if (logHistory.length > maxHistory) {
    logHistory.shift();
  }
}

// ============================================================================
// Logging Interface
// ============================================================================


export const log = {
  
  debug(module, message, data) {
    if (!shouldLog(module, LOG_LEVELS.DEBUG)) return;
    if (emitDiagnostic({ level: 'DEBUG', module, message, data })) return;
    const formatted = formatMessage(module, message);
    storeLog('DEBUG', module, message, data);
    if (data !== undefined) {
      console.debug(formatted, data);
    } else {
      console.debug(formatted);
    }
  },

  
  verbose(module, message, data) {
    if (!shouldLog(module, LOG_LEVELS.VERBOSE)) return;
    if (emitDiagnostic({ level: 'VERBOSE', module, message, data })) return;
    const formatted = formatMessage(module, message);
    storeLog('VERBOSE', module, message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  
  info(module, message, data) {
    if (!shouldLog(module, LOG_LEVELS.INFO)) return;
    if (emitDiagnostic({ level: 'INFO', module, message, data })) return;
    const formatted = formatMessage(module, message);
    storeLog('INFO', module, message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  
  warn(module, message, data) {
    if (!shouldLog(module, LOG_LEVELS.WARN)) return;
    if (emitDiagnostic({ level: 'WARN', module, message, data })) return;
    const formatted = formatMessage(module, message);
    storeLog('WARN', module, message, data);
    if (data !== undefined) {
      console.warn(formatted, data);
    } else {
      console.warn(formatted);
    }
  },

  
  error(module, message, data) {
    if (!shouldLog(module, LOG_LEVELS.ERROR)) return;
    if (emitDiagnostic({ level: 'ERROR', module, message, data })) return;
    const formatted = formatMessage(module, message);
    storeLog('ERROR', module, message, data);
    if (data !== undefined) {
      console.error(formatted, data);
    } else {
      console.error(formatted);
    }
  },

  
  always(module, message, data) {
    if (emitDiagnostic({ level: 'ALWAYS', module, message, data })) return;
    const formatted = formatMessage(module, message);
    storeLog('ALWAYS', module, message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },
};
