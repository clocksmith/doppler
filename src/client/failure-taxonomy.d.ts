/**
 * Classify a provider failure into a normalized FailureRecord for receipts
 * and diagnostics. Single taxonomy; `failureClass` values match the
 * `FailureClass` type exported by provider.d.ts.
 *
 * @param {Error|unknown} error
 * @param {{
 *   stage?: string,
 *   surface?: string,
 *   device?: string|null,
 *   modelId?: string|null,
 *   runtimeProfile?: string|null,
 *   kernelPathId?: string|null,
 * }} [context]
 * @returns {{
 *   failureClass: string,
 *   failureCode: string,
 *   stage: string,
 *   surface: string,
 *   device: string|null,
 *   modelId: string|null,
 *   runtimeProfile: string|null,
 *   kernelPathId: string|null,
 *   isSimulated: boolean,
 *   message: string,
 * }}
 */
export function classifyProviderFailure(error: Error | unknown, context?: {
    stage?: string;
    surface?: string;
    device?: string | null;
    modelId?: string | null;
    runtimeProfile?: string | null;
    kernelPathId?: string | null;
}): {
    failureClass: string;
    failureCode: string;
    stage: string;
    surface: string;
    device: string | null;
    modelId: string | null;
    runtimeProfile: string | null;
    kernelPathId: string | null;
    isSimulated: boolean;
    message: string;
};
/** Maps a legacy snake_case failure class (gpu_oom) to its kebab-case form; other values pass through. */
export function normalizeFailureClass<T>(value: T): T | string;
export const FAILURE_CLASSES: Readonly<{
    GPU_DEVICE_LOST: "gpu-device-lost";
    GPU_OOM: "gpu-oom";
    GPU_TIMEOUT: "gpu-timeout";
    GPU_UNSUPPORTED: "gpu-unsupported";
    GPU_UNAVAILABLE: "gpu-unavailable";
    MODEL_LOAD_FAILED: "model-load-failed";
    POLICY_DENIED: "policy-denied";
    RUNTIME_INTERNAL: "runtime-internal";
    FALLBACK_FAILED: "fallback-failed";
    UNKNOWN: "unknown";
}>;
