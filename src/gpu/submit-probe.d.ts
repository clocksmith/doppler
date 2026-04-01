/**
 * GPU submit latency probe.
 *
 * Dispatches a trivial compute shader, submits, waits for the GPU fence
 * plus a staging-buffer readback, and returns the wall-clock roundtrip in ms.
 */

/**
 * Run a single submit+readback roundtrip and return the elapsed time in ms.
 * Returns `null` if the probe cannot run.
 */
export function probeSubmitLatency(device: GPUDevice): Promise<number | null>;
