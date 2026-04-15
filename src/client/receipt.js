/**
 * @typedef {'local' | 'fallback'} InferenceSource
 *
 * @typedef {Object} ReceiptModel
 * @property {string} id
 * @property {string|null} hash
 * @property {string|null} fallbackId
 *
 * @typedef {Object} ReceiptDevice
 * @property {string} vendor
 * @property {string} architecture
 * @property {string} device
 * @property {string} description
 * @property {boolean} hasF16
 * @property {boolean} hasSubgroups
 * @property {number} maxBufferSize
 * @property {number|null} submitProbeMs
 * @property {number} deviceEpoch
 *
 * @typedef {Object} ReceiptFailure
 * @property {string} failureClass
 * @property {string} failureCode
 * @property {string} stage
 * @property {string} surface
 * @property {string|null} device
 * @property {string|null} modelId
 * @property {string|null} runtimeProfile
 * @property {string|null} kernelPathId
 * @property {boolean} isSimulated
 * @property {string} message
 *
 * @typedef {Object} ReceiptFallbackDecision
 * @property {string} reason
 * @property {boolean} eligible
 * @property {boolean} executed
 * @property {string|null} deniedReason
 *
 * @typedef {Object} ProviderReceiptV1
 * @property {'doppler_provider_receipt_v1'} receiptVersion
 * @property {string} receiptId
 * @property {InferenceSource} source
 * @property {string} policyMode
 * @property {string|null} policyId
 * @property {ReceiptModel} model
 * @property {ReceiptDevice|null} device
 * @property {ReceiptFailure|null} failure
 * @property {ReceiptFallbackDecision|null} fallbackDecision
 * @property {number|null} localDurationMs
 * @property {number|null} fallbackDurationMs
 * @property {number} totalDurationMs
 * @property {string} timestamp
 * @property {string|null} diagnoseArtifactRef
 */

function safeRandomUUID() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  // Fallback for environments without crypto.randomUUID
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function buildDeviceSnapshot(deviceInfo, kernelCapabilities, deviceEpoch) {
  if (!deviceInfo && !kernelCapabilities) return null;
  const info = deviceInfo || kernelCapabilities?.adapterInfo || {};
  return {
    vendor: String(info.vendor || 'unknown'),
    architecture: String(info.architecture || 'unknown'),
    device: String(info.device || 'unknown'),
    description: String(info.description || ''),
    hasF16: Boolean(kernelCapabilities?.hasF16),
    hasSubgroups: Boolean(kernelCapabilities?.hasSubgroups),
    maxBufferSize: Number(kernelCapabilities?.maxBufferSize || 0),
    submitProbeMs: kernelCapabilities?.submitProbeMs ?? null,
    deviceEpoch: Number(deviceEpoch || 0),
  };
}

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
 * @returns {ProviderReceiptV1}
 */
export function buildProviderReceiptV1({
  source,
  policyMode,
  policyId = null,
  model = {},
  deviceInfo = null,
  kernelCapabilities = null,
  deviceEpoch = 0,
  failure = null,
  fallbackDecision = null,
  localDurationMs = null,
  fallbackDurationMs = null,
  totalDurationMs,
  diagnoseArtifactRef = null,
}) {
  return {
    receiptVersion: 'doppler_provider_receipt_v1',
    receiptId: safeRandomUUID(),
    source,
    policyMode: String(policyMode || ''),
    policyId: policyId ? String(policyId) : null,
    model: {
      id: String(model.id || ''),
      hash: model.hash ? String(model.hash) : null,
      fallbackId: model.fallbackId ? String(model.fallbackId) : null,
    },
    device: buildDeviceSnapshot(deviceInfo, kernelCapabilities, deviceEpoch),
    failure: failure
      ? {
        failureClass: String(failure.failureClass || 'unknown'),
        failureCode: String(failure.failureCode || ''),
        stage: String(failure.stage || 'unknown'),
        surface: String(failure.surface || 'unknown'),
        device: failure.device ? String(failure.device) : null,
        modelId: failure.modelId ? String(failure.modelId) : null,
        runtimeProfile: failure.runtimeProfile ? String(failure.runtimeProfile) : null,
        kernelPathId: failure.kernelPathId ? String(failure.kernelPathId) : null,
        isSimulated: Boolean(failure.isSimulated),
        message: String(failure.message || ''),
      }
      : null,
    fallbackDecision: fallbackDecision
      ? {
        reason: String(fallbackDecision.reason || ''),
        eligible: Boolean(fallbackDecision.eligible),
        executed: Boolean(fallbackDecision.executed),
        deniedReason: fallbackDecision.deniedReason ? String(fallbackDecision.deniedReason) : null,
      }
      : null,
    localDurationMs: typeof localDurationMs === 'number' ? localDurationMs : null,
    fallbackDurationMs: typeof fallbackDurationMs === 'number' ? fallbackDurationMs : null,
    totalDurationMs: Number(totalDurationMs || 0),
    timestamp: new Date().toISOString(),
    diagnoseArtifactRef: diagnoseArtifactRef ? String(diagnoseArtifactRef) : null,
  };
}
