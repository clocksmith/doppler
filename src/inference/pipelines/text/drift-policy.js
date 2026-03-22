
// Drift policy for operator-level differential debugging.
//
// Loads per-operator-class tolerances from structured JSON config.
// Provides tolerance lookup for first-divergence detection.
// Models propagation bounds to prevent unbounded drift amplification.

import driftPoliciesConfig from '../../../config/runtime/diagnostics/drift-policies.json' with { type: 'json' };

// ============================================================================
// Policy Lookup
// ============================================================================

const POLICIES = driftPoliciesConfig.policies;
const PROPAGATION = driftPoliciesConfig.propagation;

export function getDriftTolerance(operatorClass, precisionMode) {
  if (!operatorClass) return null;

  const policy = POLICIES[operatorClass];
  if (!policy) return null;

  const mode = normalizePrecisionMode(precisionMode);
  const tolerance = policy.tolerances[mode];
  if (!tolerance) return null;

  return {
    maxAbsDiff: tolerance.maxAbsDiff,
    maxRelDiff: tolerance.maxRelDiff,
    propagationWeight: policy.propagationWeight,
  };
}

export function getDriftPolicyId(operatorClass) {
  if (!operatorClass) return null;
  return operatorClass in POLICIES ? operatorClass : null;
}

export function getOperatorClasses() {
  return Object.keys(POLICIES);
}

// ============================================================================
// Precision Mode Normalization
// ============================================================================

function normalizePrecisionMode(mode) {
  if (!mode || typeof mode !== 'string') return 'f32';
  const lower = mode.toLowerCase();
  if (lower === 'f32' || lower === 'float32') return 'f32';
  if (lower === 'f16' || lower === 'float16') return 'f16';
  if (lower.startsWith('q4') || lower.startsWith('q8') || lower.includes('quant')) return 'q4k';
  return 'f32';
}

// ============================================================================
// Drift Checking
// ============================================================================

export function checkDrift(operatorClass, precisionMode, observed) {
  const tolerance = getDriftTolerance(operatorClass, precisionMode);
  if (!tolerance) {
    return { withinBudget: true, reason: 'no_policy' };
  }

  const { maxAbsDiff, maxRelDiff } = tolerance;
  const absDiff = observed.maxAbsDiff ?? 0;
  const relDiff = observed.maxRelDiff ?? 0;

  const absExceeded = absDiff > maxAbsDiff;
  const relExceeded = relDiff > maxRelDiff;

  if (!absExceeded && !relExceeded) {
    return { withinBudget: true, reason: 'within_tolerance' };
  }

  return {
    withinBudget: false,
    reason: absExceeded && relExceeded ? 'both_exceeded'
      : absExceeded ? 'abs_exceeded'
        : 'rel_exceeded',
    tolerance,
    observed: { maxAbsDiff: absDiff, maxRelDiff: relDiff },
  };
}

// ============================================================================
// Propagation Bound Checking
// ============================================================================

export function checkPropagationBound(precisionMode, layerDrifts) {
  const mode = normalizePrecisionMode(precisionMode);
  const maxAccumulated = PROPAGATION.maxAccumulatedDrift[mode] ?? PROPAGATION.maxAccumulatedDrift.f32;
  const amplificationThreshold = PROPAGATION.amplificationThreshold;
  const windowSize = PROPAGATION.windowSize;

  if (!layerDrifts || layerDrifts.length === 0) {
    return { withinBound: true, reason: 'no_data' };
  }

  let accumulated = 0;
  for (const drift of layerDrifts) {
    accumulated += (drift.maxAbsDiff ?? 0) * (drift.propagationWeight ?? 1.0);
  }

  if (accumulated > maxAccumulated) {
    return {
      withinBound: false,
      reason: 'accumulated_drift_exceeded',
      accumulated,
      limit: maxAccumulated,
    };
  }

  if (layerDrifts.length >= windowSize) {
    const recent = layerDrifts.slice(-windowSize);
    const older = layerDrifts.slice(-windowSize * 2, -windowSize);
    if (older.length >= windowSize) {
      const recentAvg = recent.reduce((s, d) => s + (d.maxAbsDiff ?? 0), 0) / recent.length;
      const olderAvg = older.reduce((s, d) => s + (d.maxAbsDiff ?? 0), 0) / older.length;
      if (olderAvg > 0 && recentAvg / olderAvg > amplificationThreshold) {
        return {
          withinBound: false,
          reason: 'amplification_detected',
          amplification: recentAvg / olderAvg,
          threshold: amplificationThreshold,
        };
      }
    }
  }

  return { withinBound: true, reason: 'within_bound' };
}
