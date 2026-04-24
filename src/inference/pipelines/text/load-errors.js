import { getLastDeviceLossInfo } from '../../../gpu/device.js';
import { createDopplerError, ERROR_CODES } from '../../../errors/doppler-error.js';

const DEVICE_LIFECYCLE_PATTERNS = [
  /device not initialized/i,
  /gpu device not initialized/i,
  /no gpu device available/i,
];

function matchesDeviceLifecycleFailure(message) {
  return DEVICE_LIFECYCLE_PATTERNS.some((pattern) => pattern.test(message));
}

export function annotateWeightLoadError(error, details = {}) {
  if (!error || typeof error !== 'object') {
    return error;
  }
  const existingDetails = error.details && typeof error.details === 'object'
    ? error.details
    : {};
  error.details = {
    ...existingDetails,
    weightLoadFailure: {
      ...(existingDetails.weightLoadFailure && typeof existingDetails.weightLoadFailure === 'object'
        ? existingDetails.weightLoadFailure
        : {}),
      ...details,
    },
  };
  return error;
}

export function rewriteWeightLoadError(error, context = {}) {
  const message = error?.message || String(error);
  if (!matchesDeviceLifecycleFailure(message)) {
    return error;
  }

  const modelId = typeof context.modelId === 'string' && context.modelId.trim()
    ? context.modelId.trim()
    : 'unknown';
  const lastDeviceLoss = context.deviceLossInfo ?? getLastDeviceLossInfo();
  const wrapped = createDopplerError(
    ERROR_CODES.GPU_DEVICE_FAILED,
    `Weight load lost access to the WebGPU device for model "${modelId}". ` +
    'This is a device lifecycle failure during loadWeights, not proof of VRAM-capacity exhaustion. ' +
    'Check device initialization/loss logs and the first failing buffer allocation or upload.'
  );
  wrapped.cause = error;
  wrapped.details = {
    ...(error?.details && typeof error.details === 'object' ? error.details : {}),
    loadPhase: 'loadWeights',
    modelId,
    lifecycleFailure: 'device_unavailable_during_weight_load',
    originalMessage: message,
    lastDeviceLoss: lastDeviceLoss ? { ...lastDeviceLoss } : null,
  };
  return wrapped;
}
