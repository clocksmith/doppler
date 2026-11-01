
// Policy-driven tensor capture for operator-level differential debugging.
//
// Replaces ad-hoc probing with structured capture levels:
//   none     — no capture (default, zero overhead)
//   metadata — shape, dtype, stats only (min/max/mean/std, NaN/Inf counts)
//   slice    — metadata + sampled tensor elements (configurable count)
//   full     — complete buffer readback (expensive, use only at suspected divergence)
//
// Capture policy is resolved per-operator based on configuration.
// The engine can escalate detail around suspected first divergence
// without capturing everything.

import { computeArrayStats } from './stats.js';

// ============================================================================
// Capture Levels
// ============================================================================

export const CAPTURE_LEVELS = Object.freeze({
  NONE: 'none',
  METADATA: 'metadata',
  SLICE: 'slice',
  FULL: 'full',
});

const LEVEL_PRIORITY = { none: 0, metadata: 1, slice: 2, full: 3 };

// ============================================================================
// Capture Policy Resolution
// ============================================================================

// operatorClass is the canonical class of the op's stage (stage-names.js getOperatorClass),
// supplied by the caller; it is never inferred from the opId text.
export function resolveCapturePolicy(opId, config, operatorClass = null) {
  if (config === null) return CAPTURE_LEVELS.NONE;
  validateCaptureConfig(config);
  if (!config.enabled) return CAPTURE_LEVELS.NONE;

  const globalLevel = config.defaultLevel;

  if (config.targetOpIds && config.targetOpIds.length > 0) {
    if (config.targetOpIds.includes(opId)) {
      return config.targetLevel;
    }
  }

  if (config.targetOperatorClasses && config.targetOperatorClasses.length > 0) {
    if (typeof operatorClass !== 'string' || !operatorClass) {
      throw new Error(`[CapturePolicy] Operator-class filtering for "${opId}" requires its canonical operator class.`);
    }
    if (config.targetOperatorClasses.includes(operatorClass)) {
      return config.targetLevel;
    }
  }

  if (config.targetLayers && config.targetLayers.length > 0) {
    const layerIdx = extractLayerFromOpId(opId);
    if (layerIdx !== null && config.targetLayers.includes(layerIdx)) {
      return config.targetLevel;
    }
  }

  return globalLevel;
}

export function escalateCaptureLevel(current, target) {
  const currentPriority = LEVEL_PRIORITY[current] ?? 0;
  const targetPriority = LEVEL_PRIORITY[target] ?? 0;
  return targetPriority > currentPriority ? target : current;
}

// ============================================================================
// Capture Artifact Construction
// ============================================================================

export function buildCaptureArtifact(opId, level, data, options = {}) {
  const artifact = {
    opId,
    level,
    timestamp: Date.now(),
    shape: options.shape ?? null,
    dtype: options.dtype ?? null,
  };

  if (level === CAPTURE_LEVELS.NONE) {
    return artifact;
  }

  if (level === CAPTURE_LEVELS.METADATA || level === CAPTURE_LEVELS.SLICE || level === CAPTURE_LEVELS.FULL) {
    const stats = data ? computeArrayStats(data) : null;
    artifact.stats = stats ? {
      min: stats.min,
      max: stats.max,
      mean: stats.mean,
      std: stats.std,
      nanCount: stats.nanCount,
      infCount: stats.infCount,
      zeroCount: stats.zeroCount,
      elementCount: data.length,
    } : null;
  }

  if (level === CAPTURE_LEVELS.SLICE) {
    const sampleCount = options.sampleCount ?? 16;
    artifact.sample = data ? sampleElements(data, sampleCount) : null;
  }

  if (level === CAPTURE_LEVELS.FULL) {
    artifact.data = data ? Array.from(data) : null;
  }

  return artifact;
}

// ============================================================================
// Escalation Policy
// ============================================================================

export function createEscalationPolicy(options = {}) {
  const windowBefore = options.windowBefore ?? 2;
  const windowAfter = options.windowAfter ?? 1;

  return {
    windowBefore,
    windowAfter,
    baseLevel: options.baseLevel ?? CAPTURE_LEVELS.METADATA,
    escalatedLevel: options.escalatedLevel ?? CAPTURE_LEVELS.FULL,

    resolveForIndex(opIndex, suspectedDivergenceIndex) {
      if (suspectedDivergenceIndex === null || suspectedDivergenceIndex === undefined) {
        return this.baseLevel;
      }
      const lower = suspectedDivergenceIndex - windowBefore;
      const upper = suspectedDivergenceIndex + windowAfter;
      if (opIndex >= lower && opIndex <= upper) {
        return this.escalatedLevel;
      }
      return this.baseLevel;
    },
  };
}

// ============================================================================
// Capture Config Schema
// ============================================================================

export function createDefaultCaptureConfig() {
  return {
    enabled: false,
    defaultLevel: CAPTURE_LEVELS.NONE,
    targetLevel: CAPTURE_LEVELS.SLICE,
    targetOpIds: [],
    targetOperatorClasses: [],
    targetLayers: [],
    sampleCount: 16,
    escalation: null,
  };
}

export function validateCaptureConfig(config) {
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error('[CapturePolicy] config must be an object.');
  }

  requireCaptureLevel(config.defaultLevel, 'defaultLevel');
  requireCaptureLevel(config.targetLevel, 'targetLevel');
  if (typeof config.enabled !== 'boolean') {
    throw new Error('[CapturePolicy] enabled must be a boolean.');
  }
  for (const key of ['targetOpIds', 'targetOperatorClasses', 'targetLayers']) {
    const values = config[key];
    const validEntry = key === 'targetLayers'
      ? (entry) => Number.isSafeInteger(entry) && entry >= 0
      : (entry) => typeof entry === 'string' && entry.trim().length > 0;
    if (!Array.isArray(values) || !Array.from(values).every(validEntry)) {
      throw new Error(`[CapturePolicy] ${key} must be an array of ${key === 'targetLayers' ? 'non-negative integers' : 'non-empty strings'}.`);
    }
  }
  requireCaptureCount(config.sampleCount, 'sampleCount');
  if (config.escalation !== null) {
    const escalation = config.escalation;
    if (!escalation || typeof escalation !== 'object' || Array.isArray(escalation)) {
      throw new Error('[CapturePolicy] escalation must be an escalation policy or null.');
    }
    requireCaptureCount(escalation.windowBefore, 'escalation.windowBefore');
    requireCaptureCount(escalation.windowAfter, 'escalation.windowAfter');
    requireCaptureLevel(escalation.baseLevel, 'escalation.baseLevel');
    requireCaptureLevel(escalation.escalatedLevel, 'escalation.escalatedLevel');
    if (typeof escalation.resolveForIndex !== 'function') {
      throw new Error('[CapturePolicy] escalation.resolveForIndex must be a function.');
    }
  }
  return true;
}

function requireCaptureLevel(level, label) {
  if (typeof level !== 'string' || !Object.hasOwn(LEVEL_PRIORITY, level)) {
    throw new Error(`[CapturePolicy] Invalid ${label}: "${level}". Must be one of: ${Object.keys(LEVEL_PRIORITY).join(', ')}.`);
  }
}

function requireCaptureCount(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`[CapturePolicy] ${label} must be a non-negative integer.`);
  }
}

// ============================================================================
// Internal Helpers
// ============================================================================

function sampleElements(data, count) {
  if (data.length <= count) return Array.from(data);
  const result = [];
  const step = Math.max(1, Math.floor(data.length / count));
  for (let i = 0; i < data.length && result.length < count; i += step) {
    result.push(data[i]);
  }
  return result;
}

function extractLayerFromOpId(opId) {
  const match = opId.match(/^layer\.(\d+)\./);
  return match ? parseInt(match[1], 10) : null;
}

