
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

function normalizeResolutionIdentity(resolution) {
  if (resolution == null) return null;
  const normalized = {
    schema: String(resolution.schema || ''),
    logicalModelId: String(resolution.logicalModelId || '').trim(),
    resolvedArtifactVariantId: String(resolution.resolvedArtifactVariantId || '').toLowerCase(),
    resolvedExecutionId: String(resolution.resolvedExecutionId || '').toLowerCase(),
  };
  if (normalized.schema !== 'doppler.resolution-identity/v1') {
    throw new Error('Provider receipt resolution must use doppler.resolution-identity/v1.');
  }
  if (!normalized.logicalModelId) {
    throw new Error('Provider receipt resolution requires logicalModelId.');
  }
  for (const field of ['resolvedArtifactVariantId', 'resolvedExecutionId']) {
    if (!/^sha256:[0-9a-f]{64}$/.test(normalized[field])) {
      throw new Error(`Provider receipt resolution requires ${field} as a SHA-256 digest.`);
    }
  }
  return normalized;
}

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
  resolution = null,
  resolutionUnavailableReason = null,
}) {
  const resolvedIdentity = normalizeResolutionIdentity(resolution);
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
    resolutionStatus: resolvedIdentity ? 'resolved' : 'unavailable',
    resolution: resolvedIdentity,
    resolutionUnavailableReason: resolvedIdentity
      ? null
      : String(resolutionUnavailableReason || 'not-provided'),
  };
}
