export const ELECTRON_RELEASE_IPC_CHANNEL = 'doppler:release:v1';

const ACTION_FIELDS = Object.freeze({
  status: [],
  'resolve-current': [],
  'install-candidate': ['capsule', 'decisionDigest'],
  activate: ['decision', 'customerAuthorizationDigest'],
  reject: ['failureBundleDigest'],
  rollback: ['customerAuthorizationDigest'],
  'apply-revocations': ['snapshot'],
});

export function validateElectronReleaseIpcRequest(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Electron release IPC request must be an object.');
  }
  if (!Object.hasOwn(ACTION_FIELDS, value.action)) {
    throw new Error(`Electron release IPC action "${value.action || ''}" is unsupported.`);
  }
  const fields = new Set(['action', ...ACTION_FIELDS[value.action]]);
  for (const field of Object.keys(value)) {
    if (!fields.has(field)) throw new Error(`Electron release IPC request.${field} is unsupported.`);
  }
  for (const field of fields) {
    if (!Object.hasOwn(value, field)) throw new Error(`Electron release IPC request.${field} is required.`);
  }
  return structuredClone(value);
}

export function createElectronReleaseIpcHandler(coordinator, options) {
  if (!coordinator || typeof coordinator.load !== 'function') {
    throw new Error('Electron release IPC handler requires a release-state coordinator.');
  }
  if (typeof options?.authorizeRequest !== 'function') {
    throw new Error('Electron release IPC handler requires application-owned authorizeRequest().');
  }
  return async (event, input) => {
    const request = validateElectronReleaseIpcRequest(input);
    if (await options.authorizeRequest(event, structuredClone(request)) !== true) {
      const error = new Error('Electron release IPC request is not authorized by the application.');
      error.code = 'DOPPLER_ELECTRON_UNAUTHORIZED';
      throw error;
    }
    if (request.action === 'status') return coordinator.load();
    if (request.action === 'resolve-current') return coordinator.resolveCurrent();
    if (request.action === 'install-candidate') {
      return coordinator.installCandidate(request.capsule, request.decisionDigest);
    }
    if (request.action === 'activate') {
      return coordinator.activateCandidate(request.decision, request.customerAuthorizationDigest);
    }
    if (request.action === 'reject') return coordinator.rejectCandidate(request.failureBundleDigest);
    if (request.action === 'rollback') return coordinator.rollback(request.customerAuthorizationDigest);
    return coordinator.applyRevocationSnapshot(request.snapshot);
  };
}
