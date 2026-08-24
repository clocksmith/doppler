/**
 * Build a structured v1 provider receipt.
 *
 * @param {Object} params
 * @param {InferenceSource} params.source
 * @param {string} params.policyMode
 * @param {string|null} [params.policyId]
 * @param {{ id?: string, hash?: string|null, fallbackId?: string|null }} [params.model]
 * @param {Object|null} [params.deviceInfo]
 * @param {Object|null} [params.kernelCapabilities]
 * @param {number} [params.deviceEpoch]
 * @param {import('./failure-taxonomy.js').classifyProviderFailure|null} [params.failure]
 * @param {{ reason?: string, eligible?: boolean, executed?: boolean, deniedReason?: string|null }|null} [params.fallbackDecision]
 * @param {number|null} [params.localDurationMs]
 * @param {number|null} [params.fallbackDurationMs]
 * @param {number} params.totalDurationMs
 * @param {string|null} [params.diagnoseArtifactRef]
 * @param {Object|null} [params.resolution]
 * @param {string|null} [params.resolutionUnavailableReason]
 * @returns {ProviderReceiptV1}
 */
export function buildProviderReceiptV1({ source, policyMode, policyId, model, deviceInfo, kernelCapabilities, deviceEpoch, failure, fallbackDecision, localDurationMs, fallbackDurationMs, totalDurationMs, diagnoseArtifactRef, resolution, resolutionUnavailableReason, }: {
    source: InferenceSource;
    policyMode: string;
    policyId?: string | null | undefined;
    model?: {
        id?: string;
        hash?: string | null;
        fallbackId?: string | null;
    } | undefined;
    deviceInfo?: Object | null | undefined;
    kernelCapabilities?: Object | null | undefined;
    deviceEpoch?: number | undefined;
    failure?: typeof import("./failure-taxonomy.js").classifyProviderFailure | null | undefined;
    fallbackDecision?: {
        reason?: string;
        eligible?: boolean;
        executed?: boolean;
        deniedReason?: string | null;
    } | null | undefined;
    localDurationMs?: number | null | undefined;
    fallbackDurationMs?: number | null | undefined;
    totalDurationMs: number;
    diagnoseArtifactRef?: string | null | undefined;
    resolution?: Object | null | undefined;
    resolutionUnavailableReason?: string | null | undefined;
}): ProviderReceiptV1;
export type InferenceSource = "local" | "fallback";
export type ReceiptModel = {
    id: string;
    hash: string | null;
    fallbackId: string | null;
};
export type ReceiptDevice = {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
    hasF16: boolean;
    hasSubgroups: boolean;
    maxBufferSize: number;
    submitProbeMs: number | null;
    deviceEpoch: number;
};
export type ReceiptFailure = {
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
export type ReceiptFallbackDecision = {
    reason: string;
    eligible: boolean;
    executed: boolean;
    deniedReason: string | null;
};
export type ProviderReceiptV1 = {
    receiptVersion: "doppler_provider_receipt_v1";
    receiptId: string;
    source: InferenceSource;
    policyMode: string;
    policyId: string | null;
    model: ReceiptModel;
    device: ReceiptDevice | null;
    failure: ReceiptFailure | null;
    fallbackDecision: ReceiptFallbackDecision | null;
    localDurationMs: number | null;
    fallbackDurationMs: number | null;
    totalDurationMs: number;
    timestamp: string;
    diagnoseArtifactRef: string | null;
    resolutionStatus: "resolved" | "unavailable";
    resolution: {
        schema: "doppler.resolution-identity/v1";
        logicalModelId: string;
        resolvedArtifactVariantId: string;
        resolvedExecutionId: string;
    } | null;
    resolutionUnavailableReason: string | null;
};
