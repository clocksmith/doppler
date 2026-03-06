export { DOPPLER_PROVIDER_VERSION } from '../../version.js';

export const DopplerCapabilities = {
  available: false,
  HAS_MEMORY64: false,
  HAS_SUBGROUPS: false,
  HAS_F16: false,
  IS_UNIFIED_MEMORY: false,
  TIER_LEVEL: 1,
  TIER_NAME: '',
  MAX_MODEL_SIZE: 0,
  initialized: false,
  currentModelId: null,
  kernelsWarmed: false,
  kernelsTuned: false,
  lastModelEstimate: null,
};
