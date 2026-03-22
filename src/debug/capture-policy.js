
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

export function resolveCapturePolicy(opId, config) {
  if (!config || !config.enabled) return CAPTURE_LEVELS.NONE;

  const globalLevel = config.defaultLevel ?? CAPTURE_LEVELS.NONE;

  if (config.targetOpIds && config.targetOpIds.length > 0) {
    if (config.targetOpIds.includes(opId)) {
      return config.targetLevel ?? CAPTURE_LEVELS.FULL;
    }
  }

  if (config.targetOperatorClasses && config.targetOperatorClasses.length > 0) {
    const opClass = extractOperatorClassFromOpId(opId);
    if (opClass && config.targetOperatorClasses.includes(opClass)) {
      return config.targetLevel ?? CAPTURE_LEVELS.SLICE;
    }
  }

  if (config.targetLayers && config.targetLayers.length > 0) {
    const layerIdx = extractLayerFromOpId(opId);
    if (layerIdx !== null && config.targetLayers.includes(layerIdx)) {
      return config.targetLevel ?? CAPTURE_LEVELS.SLICE;
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
  if (!config || typeof config !== 'object') {
    throw new Error('[CapturePolicy] config must be an object.');
  }

  const level = config.defaultLevel;
  if (level && !(level in LEVEL_PRIORITY)) {
    throw new Error(`[CapturePolicy] Invalid defaultLevel: "${level}". Must be one of: ${Object.keys(LEVEL_PRIORITY).join(', ')}.`);
  }

  const targetLevel = config.targetLevel;
  if (targetLevel && !(targetLevel in LEVEL_PRIORITY)) {
    throw new Error(`[CapturePolicy] Invalid targetLevel: "${targetLevel}". Must be one of: ${Object.keys(LEVEL_PRIORITY).join(', ')}.`);
  }

  return true;
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

function extractOperatorClassFromOpId(opId) {
  const withoutLayer = opId.replace(/^layer\.\d+\./, '');
  const section = withoutLayer.split('.')[0];
  return section || null;
}
