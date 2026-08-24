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
export const FAILURE_CLASSES: Readonly<{
    GPU_DEVICE_LOST: "gpu_device_lost";
    GPU_OOM: "gpu_oom";
    GPU_TIMEOUT: "gpu_timeout";
    GPU_UNSUPPORTED: "gpu_unsupported";
    GPU_UNAVAILABLE: "gpu_unavailable";
    MODEL_LOAD_FAILED: "model_load_failed";
    POLICY_DENIED: "policy_denied";
    RUNTIME_INTERNAL: "runtime_internal";
    FALLBACK_FAILED: "fallback_failed";
    UNKNOWN: "unknown";
}>;
