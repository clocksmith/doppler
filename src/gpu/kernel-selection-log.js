import { log } from '../debug/index.js';

const loggedSelections = new Set();

export function logKernelSelectionOnce(operation, payload) {
  const key = `${operation}:${payload.variant ?? 'unknown'}`;
  if (loggedSelections.has(key)) {
    return;
  }
  loggedSelections.add(key);
  const reason = payload.reason ? ` reason=${payload.reason}` : '';
  log.info('KernelSelect', `${operation} variant=${payload.variant ?? 'unknown'}${reason}`);
}

export function resetKernelSelectionLog() {
  loggedSelections.clear();
}
