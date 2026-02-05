export const ENGINE_ORDER = ['valu', 'alu', 'flow', 'load', 'store', 'debug'];

export const WEIGHT_KEYS = {
  height: 0,
  slack: 1,
  pressure: 2,
  age: 3,
  baseline: 4,
};

export const DEFAULT_WEIGHTS = new Float32Array([1.0, 0.6, 0.4, 0.1, 0.2]);
