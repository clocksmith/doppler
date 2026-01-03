import { log } from '../debug/index.js';

const loggedSelections = new Set();
const selectionLog = [];

export function logKernelSelectionOnce(operation, payload) {
  const key = `${operation}:${payload.variant ?? 'unknown'}`;
  if (loggedSelections.has(key)) {
    return;
  }
  loggedSelections.add(key);
  selectionLog.push({
    operation,
    variant: payload.variant ?? 'unknown',
    reason: payload.reason ?? null,
  });
  const reason = payload.reason ? ` reason=${payload.reason}` : '';
  log.info('KernelSelect', `${operation} variant=${payload.variant ?? 'unknown'}${reason}`);
}

export function resetKernelSelectionLog() {
  loggedSelections.clear();
  selectionLog.length = 0;
}

export function getKernelSelectionLog() {
  return [...selectionLog];
}
