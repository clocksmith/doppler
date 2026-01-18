export const DEFAULT_TUNER_CONFIG = {
  cacheKeyPrefix: 'doppler_kernel_tune_',
  defaultWarmupIterations: 3,
  defaultTimedIterations: 10,
  fallbackWorkgroupSizes: {
    matmul: [16, 16, 1],
    attention: [256, 1, 1],
    rmsnorm: [256, 1, 1],
    softmax: [256, 1, 1],
    dequant: [64, 1, 1],
    default: [256, 1, 1],
  },
};
