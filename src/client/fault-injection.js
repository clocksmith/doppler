/**
 * Config-gated fault injection for the Doppler provider.
 *
 * Injected errors carry `__dopplerFaultInjected = true` so the failure
 * taxonomy marks them `isSimulated: true` in the receipt.
 *
 * @param {{ diagnostics?: { faultInjection?: { enabled?: boolean, failureCode?: string, stage?: string, probability?: number } } }} config
 */
export function createFaultInjector(config) {
  const fi = config?.diagnostics?.faultInjection;
  const enabled = fi?.enabled === true;
  const probability = typeof fi?.probability === 'number' ? Math.max(0, Math.min(1, fi.probability)) : 1;
  const failureCode = String(fi?.failureCode || 'DOPPLER_GPU_OOM');
  const stage = String(fi?.stage || 'generate');

  function shouldInject(currentStage) {
    if (!enabled) return false;
    if (stage !== '*' && stage !== currentStage) return false;
    return Math.random() < probability;
  }

  function createInjectedError() {
    const error = new Error(`[fault-injection] Simulated failure: ${failureCode}`);
    error.__dopplerFaultInjected = true;
    error.code = failureCode;
    return error;
  }

  return { shouldInject, createInjectedError };
}
