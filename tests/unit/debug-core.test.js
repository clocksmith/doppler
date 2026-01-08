import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';

import {
  LOG_LEVELS,
  TRACE_CATEGORIES,
  setLogLevel,
  getLogLevel,
  setTrace,
  getTrace,
  isTraceEnabled,
  incrementDecodeStep,
  resetDecodeStep,
  getDecodeStep,
  setBenchmarkMode,
  isBenchmarkMode,
  enableModules,
  disableModules,
  resetModuleFilters,
  getLogHistory,
  clearLogHistory,
  getDebugSnapshot,
} from '../../src/debug/index.js';

import { log } from '../../src/debug/log.js';
import { trace } from '../../src/debug/trace.js';

describe('debug/config', () => {
  beforeEach(() => {
    setLogLevel('info');
    setTrace(false);
    resetModuleFilters();
    resetDecodeStep();
    clearLogHistory();
    if (isBenchmarkMode()) {
      setBenchmarkMode(false);
    }
  });

  afterEach(() => {
    if (isBenchmarkMode()) {
      setBenchmarkMode(false);
    }
  });

  describe('LOG_LEVELS', () => {
    it('defines all standard log levels', () => {
      expect(LOG_LEVELS.DEBUG).toBe(0);
      expect(LOG_LEVELS.VERBOSE).toBe(1);
      expect(LOG_LEVELS.INFO).toBe(2);
      expect(LOG_LEVELS.WARN).toBe(3);
      expect(LOG_LEVELS.ERROR).toBe(4);
      expect(LOG_LEVELS.SILENT).toBe(5);
    });

    it('has increasing values for more restrictive levels', () => {
      expect(LOG_LEVELS.DEBUG).toBeLessThan(LOG_LEVELS.VERBOSE);
      expect(LOG_LEVELS.VERBOSE).toBeLessThan(LOG_LEVELS.INFO);
      expect(LOG_LEVELS.INFO).toBeLessThan(LOG_LEVELS.WARN);
      expect(LOG_LEVELS.WARN).toBeLessThan(LOG_LEVELS.ERROR);
      expect(LOG_LEVELS.ERROR).toBeLessThan(LOG_LEVELS.SILENT);
    });
  });

  describe('TRACE_CATEGORIES', () => {
    it('includes all expected categories', () => {
      expect(TRACE_CATEGORIES).toContain('loader');
      expect(TRACE_CATEGORIES).toContain('kernels');
      expect(TRACE_CATEGORIES).toContain('logits');
      expect(TRACE_CATEGORIES).toContain('embed');
      expect(TRACE_CATEGORIES).toContain('attn');
      expect(TRACE_CATEGORIES).toContain('ffn');
      expect(TRACE_CATEGORIES).toContain('kv');
      expect(TRACE_CATEGORIES).toContain('sample');
      expect(TRACE_CATEGORIES).toContain('buffers');
      expect(TRACE_CATEGORIES).toContain('perf');
    });

    it('has exactly 10 categories', () => {
      expect(TRACE_CATEGORIES.length).toBe(10);
    });
  });

  describe('setLogLevel / getLogLevel', () => {
    it('sets and gets log level', () => {
      setLogLevel('debug');
      expect(getLogLevel()).toBe('debug');

      setLogLevel('warn');
      expect(getLogLevel()).toBe('warn');
    });

    it('handles case-insensitive input', () => {
      setLogLevel('DEBUG');
      expect(getLogLevel()).toBe('debug');

      setLogLevel('WaRn');
      expect(getLogLevel()).toBe('warn');
    });

    it('defaults to info for unknown level', () => {
      setLogLevel('unknown');
      expect(getLogLevel()).toBe('info');
    });

    it('supports all defined levels', () => {
      const levels = ['debug', 'verbose', 'info', 'warn', 'error', 'silent'];
      for (const level of levels) {
        setLogLevel(level);
        expect(getLogLevel()).toBe(level);
      }
    });
  });

  describe('setTrace / getTrace', () => {
    it('enables single category via string', () => {
      setTrace('kernels');
      const enabled = getTrace();
      expect(enabled).toContain('kernels');
      expect(enabled.length).toBe(1);
    });

    it('enables multiple categories via comma-separated string', () => {
      setTrace('kernels,logits,attn');
      const enabled = getTrace();
      expect(enabled).toContain('kernels');
      expect(enabled).toContain('logits');
      expect(enabled).toContain('attn');
      expect(enabled.length).toBe(3);
    });

    it('enables all categories via "all"', () => {
      setTrace('all');
      const enabled = getTrace();
      expect(enabled.length).toBe(TRACE_CATEGORIES.length);
      for (const cat of TRACE_CATEGORIES) {
        expect(enabled).toContain(cat);
      }
    });

    it('excludes categories via minus prefix', () => {
      setTrace('all,-buffers,-perf');
      const enabled = getTrace();
      expect(enabled).not.toContain('buffers');
      expect(enabled).not.toContain('perf');
      expect(enabled).toContain('kernels');
      expect(enabled.length).toBe(TRACE_CATEGORIES.length - 2);
    });

    it('disables all tracing via false', () => {
      setTrace('all');
      expect(getTrace().length).toBe(TRACE_CATEGORIES.length);

      setTrace(false);
      expect(getTrace().length).toBe(0);
    });

    it('accepts array of categories', () => {
      setTrace(['kernels', 'attn', 'ffn']);
      const enabled = getTrace();
      expect(enabled).toContain('kernels');
      expect(enabled).toContain('attn');
      expect(enabled).toContain('ffn');
      expect(enabled.length).toBe(3);
    });

    it('ignores unknown categories', () => {
      setTrace('kernels,unknown,logits');
      const enabled = getTrace();
      expect(enabled).toContain('kernels');
      expect(enabled).toContain('logits');
      expect(enabled).not.toContain('unknown');
      expect(enabled.length).toBe(2);
    });

    it('handles whitespace in string', () => {
      setTrace('kernels, logits , attn');
      const enabled = getTrace();
      expect(enabled).toContain('kernels');
      expect(enabled).toContain('logits');
      expect(enabled).toContain('attn');
    });
  });

  describe('isTraceEnabled', () => {
    it('returns false when category is not enabled', () => {
      setTrace(false);
      expect(isTraceEnabled('kernels')).toBe(false);
    });

    it('returns true when category is enabled', () => {
      setTrace('kernels');
      expect(isTraceEnabled('kernels')).toBe(true);
      expect(isTraceEnabled('logits')).toBe(false);
    });

    it('respects layer filter when provided', () => {
      setTrace('attn', { layers: [0, 5, 10] });

      expect(isTraceEnabled('attn', 0)).toBe(true);
      expect(isTraceEnabled('attn', 5)).toBe(true);
      expect(isTraceEnabled('attn', 3)).toBe(false);
      expect(isTraceEnabled('attn')).toBe(true);
    });

    it('respects max decode steps limit', () => {
      setTrace('sample', { maxDecodeSteps: 5 });

      expect(isTraceEnabled('sample')).toBe(true);

      for (let i = 0; i < 6; i++) {
        incrementDecodeStep();
      }

      expect(isTraceEnabled('sample')).toBe(false);
    });
  });

  describe('decode step tracking', () => {
    it('starts at zero', () => {
      expect(getDecodeStep()).toBe(0);
    });

    it('increments correctly', () => {
      expect(incrementDecodeStep()).toBe(1);
      expect(incrementDecodeStep()).toBe(2);
      expect(incrementDecodeStep()).toBe(3);
      expect(getDecodeStep()).toBe(3);
    });

    it('resets to zero', () => {
      incrementDecodeStep();
      incrementDecodeStep();
      resetDecodeStep();
      expect(getDecodeStep()).toBe(0);
    });
  });

  describe('benchmark mode', () => {
    it('defaults to disabled', () => {
      expect(isBenchmarkMode()).toBe(false);
    });

    it('can be enabled and disabled', () => {
      setBenchmarkMode(true);
      expect(isBenchmarkMode()).toBe(true);

      setBenchmarkMode(false);
      expect(isBenchmarkMode()).toBe(false);
    });

    it('silences console methods when enabled', () => {
      const originalLog = console.log;

      setBenchmarkMode(true);
      expect(console.log).not.toBe(originalLog);

      setBenchmarkMode(false);
      expect(console.log).toBe(originalLog);
    });
  });

  describe('module filters', () => {
    it('enableModules restricts logging to specified modules', () => {
      enableModules('Pipeline', 'Loader');

      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

      log.info('Pipeline', 'Test message');
      expect(spy).toHaveBeenCalled();

      spy.mockClear();
      log.info('Other', 'Test message');
      expect(spy).not.toHaveBeenCalled();

      spy.mockRestore();
    });

    it('disableModules excludes specific modules', () => {
      disableModules('Verbose');

      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

      log.info('Pipeline', 'Test message');
      expect(spy).toHaveBeenCalled();

      spy.mockClear();
      log.info('Verbose', 'Test message');
      expect(spy).not.toHaveBeenCalled();

      spy.mockRestore();
    });

    it('resetModuleFilters clears all filters', () => {
      enableModules('Pipeline');
      disableModules('Other');

      resetModuleFilters();

      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

      log.info('Pipeline', 'Test');
      expect(spy).toHaveBeenCalled();

      spy.mockClear();
      log.info('Other', 'Test');
      expect(spy).toHaveBeenCalled();

      spy.mockRestore();
    });

    it('module filters are case-insensitive', () => {
      enableModules('PIPELINE');

      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

      log.info('pipeline', 'Test message');
      expect(spy).toHaveBeenCalled();

      spy.mockRestore();
    });
  });
});

describe('debug/log', () => {
  beforeEach(() => {
    setLogLevel('debug');
    resetModuleFilters();
    clearLogHistory();
  });

  describe('log levels', () => {
    it('log.debug outputs at debug level', () => {
      const spy = vi.spyOn(console, 'debug').mockImplementation(() => {});
      log.debug('Test', 'Debug message');
      expect(spy).toHaveBeenCalled();
      spy.mockRestore();
    });

    it('log.verbose outputs at verbose level', () => {
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
      log.verbose('Test', 'Verbose message');
      expect(spy).toHaveBeenCalled();
      spy.mockRestore();
    });

    it('log.info outputs at info level', () => {
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
      log.info('Test', 'Info message');
      expect(spy).toHaveBeenCalled();
      spy.mockRestore();
    });

    it('log.warn outputs at warn level', () => {
      const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      log.warn('Test', 'Warning message');
      expect(spy).toHaveBeenCalled();
      spy.mockRestore();
    });

    it('log.error outputs at error level', () => {
      const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
      log.error('Test', 'Error message');
      expect(spy).toHaveBeenCalled();
      spy.mockRestore();
    });

    it('log.always outputs regardless of level', () => {
      setLogLevel('silent');
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
      log.always('Test', 'Always message');
      expect(spy).toHaveBeenCalled();
      spy.mockRestore();
    });
  });

  describe('level filtering', () => {
    it('filters out messages below current level', () => {
      setLogLevel('warn');

      const debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
      const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      log.debug('Test', 'Debug');
      log.verbose('Test', 'Verbose');
      log.info('Test', 'Info');
      log.warn('Test', 'Warning');

      expect(debugSpy).not.toHaveBeenCalled();
      expect(logSpy).not.toHaveBeenCalled();
      expect(warnSpy).toHaveBeenCalled();

      debugSpy.mockRestore();
      logSpy.mockRestore();
      warnSpy.mockRestore();
    });

    it('silent level suppresses all except always', () => {
      setLogLevel('silent');

      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

      log.error('Test', 'Error');
      expect(errorSpy).not.toHaveBeenCalled();

      log.always('Test', 'Always');
      expect(logSpy).toHaveBeenCalled();

      errorSpy.mockRestore();
      logSpy.mockRestore();
    });
  });

  describe('message formatting', () => {
    it('includes timestamp and module in output', () => {
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

      log.info('Pipeline', 'Model loaded');

      expect(spy).toHaveBeenCalled();
      const message = spy.mock.calls[0][0];
      expect(message).toMatch(/\[\d+\.\d+ms\]\[Pipeline\] Model loaded/);

      spy.mockRestore();
    });

    it('includes data argument when provided', () => {
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const data = { key: 'value' };

      log.info('Test', 'Message with data', data);

      expect(spy).toHaveBeenCalledWith(expect.any(String), data);

      spy.mockRestore();
    });
  });

  describe('log history', () => {
    it('stores log entries in history', () => {
      log.info('Test', 'Entry 1');
      log.warn('Test', 'Entry 2');

      const history = getLogHistory();
      expect(history.length).toBeGreaterThanOrEqual(2);
    });

    it('log history includes level, module, and message', () => {
      log.info('TestModule', 'Test message');

      const history = getLogHistory();
      const entry = history.find(e => e.module === 'TestModule');

      expect(entry).toBeDefined();
      expect(entry.level).toBe('INFO');
      expect(entry.message).toBe('Test message');
    });

    it('clearLogHistory removes all entries', () => {
      log.info('Test', 'Entry');
      expect(getLogHistory().length).toBeGreaterThan(0);

      clearLogHistory();
      expect(getLogHistory().length).toBe(0);
    });

    it('getLogHistory supports level filter', () => {
      clearLogHistory();
      log.info('Test', 'Info');
      log.warn('Test', 'Warning');
      log.error('Test', 'Error');

      const warnings = getLogHistory({ level: 'warn' });
      expect(warnings.every(e => e.level === 'WARN')).toBe(true);
    });

    it('getLogHistory supports module filter', () => {
      clearLogHistory();
      log.info('Pipeline', 'Pipeline msg');
      log.info('Loader', 'Loader msg');

      const pipelineLogs = getLogHistory({ module: 'Pipeline' });
      expect(pipelineLogs.every(e => e.module.includes('Pipeline'))).toBe(true);
    });

    it('getLogHistory supports last N entries', () => {
      clearLogHistory();
      for (let i = 0; i < 10; i++) {
        log.info('Test', `Entry ${i}`);
      }

      const last5 = getLogHistory({ last: 5 });
      expect(last5.length).toBe(5);
    });
  });
});

describe('debug/trace', () => {
  beforeEach(() => {
    setTrace(false);
    clearLogHistory();
  });

  describe('trace categories', () => {
    it('trace.loader logs when loader category enabled', () => {
      setTrace('loader');
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

      trace.loader('Loading shard');

      expect(spy).toHaveBeenCalled();
      expect(spy.mock.calls[0][0]).toContain('[TRACE:loader]');

      spy.mockRestore();
    });

    it('trace.kernels logs when kernels category enabled', () => {
      setTrace('kernels');
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

      trace.kernels('matmul M=1 N=2048');

      expect(spy).toHaveBeenCalled();
      expect(spy.mock.calls[0][0]).toContain('[TRACE:kernels]');

      spy.mockRestore();
    });

    it('trace.attn includes layer index in output', () => {
      setTrace('attn');
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

      trace.attn(5, 'Attention computation');

      expect(spy).toHaveBeenCalled();
      expect(spy.mock.calls[0][0]).toContain('L5:');

      spy.mockRestore();
    });

    it('trace.ffn includes layer index in output', () => {
      setTrace('ffn', { layers: [] });
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

      trace.ffn(3, 'FFN forward');

      expect(spy).toHaveBeenCalled();
      expect(spy.mock.calls[0][0]).toContain('L3:');

      spy.mockRestore();
    });

    it('trace.kv includes layer index in output', () => {
      setTrace('kv', { layers: [] });
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

      trace.kv(7, 'KV cache update');

      expect(spy).toHaveBeenCalled();
      expect(spy.mock.calls[0][0]).toContain('L7:');

      spy.mockRestore();
    });
  });

  describe('trace filtering', () => {
    it('does not log when category is disabled', () => {
      setTrace(false);
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

      trace.kernels('Should not appear');
      trace.loader('Should not appear');
      trace.logits('Should not appear');

      expect(spy).not.toHaveBeenCalled();

      spy.mockRestore();
    });

    it('only logs enabled categories', () => {
      setTrace('kernels');
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

      trace.kernels('Enabled');
      expect(spy).toHaveBeenCalledTimes(1);

      trace.loader('Disabled');
      expect(spy).toHaveBeenCalledTimes(1);

      spy.mockRestore();
    });
  });

  describe('trace history', () => {
    it('stores trace entries in log history', () => {
      setTrace('kernels');
      clearLogHistory();

      trace.kernels('Kernel dispatch');

      const history = getLogHistory();
      const traceEntry = history.find(e => e.level === 'TRACE:kernels');
      expect(traceEntry).toBeDefined();
      expect(traceEntry.message).toBe('Kernel dispatch');
    });
  });
});

describe('debug/history', () => {
  beforeEach(() => {
    setLogLevel('debug');
    setTrace('all');
    clearLogHistory();
  });

  describe('getDebugSnapshot', () => {
    it('returns snapshot with current config', () => {
      setLogLevel('verbose');
      setTrace('kernels,attn');

      const snapshot = getDebugSnapshot();

      expect(snapshot.timestamp).toBeDefined();
      expect(snapshot.logLevel?.toLowerCase()).toBe('verbose');
      expect(snapshot.traceCategories).toContain('kernels');
      expect(snapshot.traceCategories).toContain('attn');
    });

    it('includes recent logs in snapshot', () => {
      log.info('Test', 'Snapshot test');
      log.error('Test', 'Error entry');

      const snapshot = getDebugSnapshot();

      expect(snapshot.recentLogs.length).toBeGreaterThan(0);
      expect(snapshot.errorCount).toBeGreaterThanOrEqual(1);
    });

    it('tracks error and warn counts', () => {
      clearLogHistory();

      log.warn('Test', 'Warning 1');
      log.warn('Test', 'Warning 2');
      log.error('Test', 'Error 1');

      const snapshot = getDebugSnapshot();

      expect(snapshot.warnCount).toBe(2);
      expect(snapshot.errorCount).toBe(1);
    });

    it('includes module filter state', () => {
      enableModules('Pipeline');
      disableModules('Verbose');

      const snapshot = getDebugSnapshot();

      expect(snapshot.enabledModules).toContain('pipeline');
      expect(snapshot.disabledModules).toContain('verbose');
    });
  });
});
