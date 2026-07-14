// ============================================================================
// Shared Experimental Tooling Surface Exports
//
// Browser-safe experimental helpers kept separate from the tier1 tooling core.
// ============================================================================

// Browser conversion + file pickers
export { convertModel, createRemoteModelSources, isConversionSupported } from './experimental/browser/browser-converter.js';
export { pickModelDirectory, pickModelFiles } from './experimental/browser/file-picker.js';

// Trainer artifact bridge contract shared by Gamma handoffs and Columbo adapters.
export {
  TRAINER_ARTIFACT_BRIDGE_SCHEMA_ID,
  TRAINER_ARTIFACT_IMPORT_PLAN_SCHEMA_ID,
  TRAINER_ARTIFACT_PARITY_EVIDENCE_SCHEMA_ID,
  TRAINER_ARTIFACT_PARITY_RECEIPT_SCHEMA_ID,
  TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT,
  TRAINER_ARTIFACT_KIND_PEFT_ADAPTER,
  TRANSLATION_FULL_CHECKPOINT_PARITY_CHECKS,
  COLUMBO_QWEN_ADAPTER_PARITY_CHECKS,
  normalizeGammaTrainerArtifactHandoff,
  validateTrainerArtifactBridgeDescriptor,
  assertTrainerArtifactCandidateEntry,
  buildTrainerArtifactImportPlan,
  buildTrainerArtifactParityTemplate,
  verifyTrainerArtifactParityEvidence,
} from './experimental/bridge/trainer-artifact-bridge.js';

// Distribution tooling
export {
  P2P_WEBRTC_DATA_PLANE_CONTRACT_VERSION,
  isBrowserWebRTCAvailable,
  createBrowserWebRTCDataPlaneTransport,
} from './experimental/distribution/p2p-webrtc-browser.js';
export {
  P2P_CONTROL_PLANE_CONTRACT_VERSION,
  normalizeP2PControlPlaneConfig,
  resolveP2PSessionToken,
  evaluateP2PPolicyDecision,
} from './experimental/distribution/p2p-control-plane.js';
export {
  P2P_OBSERVABILITY_SCHEMA_VERSION,
  createP2PDeliveryObservabilityRecord,
  aggregateP2PDeliveryObservability,
  buildP2PAlertsFromSummary,
  buildP2PDashboardSnapshot,
} from './experimental/distribution/p2p-observability.js';
export {
  DESCRIPTOR_TRANSPORT_CONTRACT_VERSION,
  normalizePeerCapabilityProfile,
  getDescriptorRequiredGenerators,
  assertPeerSupportsDescriptor,
  getDescriptorRequiredShards,
  negotiateDescriptorShardCache,
  validateActivationTransportPayload,
  createDescriptorPeerAssignment,
} from './experimental/distribution/descriptor-transport.js';
