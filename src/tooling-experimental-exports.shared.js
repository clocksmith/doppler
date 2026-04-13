// ============================================================================
// Shared Experimental Tooling Surface Exports
//
// Browser-safe experimental helpers kept separate from the tier1 tooling core.
// ============================================================================

// Browser conversion + file pickers
export { convertModel, createRemoteModelSources, isConversionSupported } from './experimental/browser/browser-converter.js';
export { pickModelDirectory, pickModelFiles } from './experimental/browser/file-picker.js';

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
