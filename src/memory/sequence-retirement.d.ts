/**
 * Retire idle pooled buffers and pending retirements after confirmed GPU
 * completion, only when the device has an explicit memory budget. Owned buffers
 * are never retired. Resolves true when a retirement boundary ran.
 */
export function retireIdleBuffersUnderDeviceBudget(device: GPUDevice | null | undefined): Promise<boolean>;
