import { performance } from 'node:perf_hooks';

export function createConversionRunTiming(options = {}) {
  const now = typeof options.now === 'function' ? options.now : () => new Date();
  const monotonicNow = typeof options.monotonicNow === 'function'
    ? options.monotonicNow
    : () => performance.now();
  const startedAtUtc = now().toISOString();
  const startedAtMs = monotonicNow();
  let completed = null;

  return {
    complete() {
      if (!completed) {
        completed = Object.freeze({
          startedAtUtc,
          completedAtUtc: now().toISOString(),
          durationMs: Math.max(0, monotonicNow() - startedAtMs),
        });
      }
      return completed;
    },
  };
}
