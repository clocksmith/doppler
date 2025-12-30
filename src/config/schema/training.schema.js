import { DEFAULT_LORA_CONFIG } from './lora.schema.js';

// =============================================================================
// Optimizer Defaults
// =============================================================================

export const DEFAULT_TRAINING_OPTIMIZER_CONFIG = {
  type: 'adam',
  lr: 2e-4,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  weightDecay: 0.0,
};

// =============================================================================
// Gradient Defaults
// =============================================================================

export const DEFAULT_TRAINING_GRADIENT_CONFIG = {
  maxNorm: 1.0,
  accumSteps: 1,
};

// =============================================================================
// Loss Scaling Defaults
// =============================================================================

export const DEFAULT_TRAINING_LOSS_SCALING_CONFIG = {
  enabled: false,
  initialScale: 1024,
  minScale: 1,
  maxScale: 65536,
  scaleFactor: 2,
  backoffFactor: 0.5,
  growthInterval: 2000,
  overflowCheck: true,
};

// =============================================================================
// Precision Defaults
// =============================================================================

export const DEFAULT_TRAINING_PRECISION_CONFIG = {
  activations: 'f16',
  gradients: 'f32',
  loraParams: 'f32',
};

// =============================================================================
// Attention Defaults
// =============================================================================

export const DEFAULT_TRAINING_ATTENTION_CONFIG = {
  recomputeForward: false,
};

// =============================================================================
// Training Defaults
// =============================================================================

export const DEFAULT_TRAINING_SETTINGS = {
  enabled: false,
  lora: DEFAULT_LORA_CONFIG,
  optimizer: DEFAULT_TRAINING_OPTIMIZER_CONFIG,
  gradient: DEFAULT_TRAINING_GRADIENT_CONFIG,
  precision: DEFAULT_TRAINING_PRECISION_CONFIG,
  attention: DEFAULT_TRAINING_ATTENTION_CONFIG,
  lossScaling: DEFAULT_TRAINING_LOSS_SCALING_CONFIG,
};
