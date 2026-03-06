function asObject(value) {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  return value;
}

function asFiniteNumber(value) {
  if (value == null || value === '') {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function asBoolean(value) {
  return typeof value === 'boolean' ? value : null;
}

function asNumberArray(value) {
  if (!Array.isArray(value)) return null;
  const normalized = value.map((entry) => asFiniteNumber(entry));
  if (normalized.some((entry) => entry == null || entry <= 0)) {
    return null;
  }
  return normalized.map((entry) => Math.trunc(entry));
}

function normalizeRoPEType(value) {
  if (typeof value !== 'string') return null;
  const normalized = value.trim().toLowerCase();
  if (normalized === '' || normalized === 'default' || normalized === 'none') {
    return null;
  }
  return normalized;
}

function resolveScalingConfig(ropeScalingConfig, options = {}) {
  const { strictMissingTypeAndFactor = false, sourceLabel = 'HF config' } = options;
  const scalingTypeRaw = ropeScalingConfig.type ?? ropeScalingConfig.rope_type;
  const scalingType = normalizeRoPEType(scalingTypeRaw);
  const factor = asFiniteNumber(ropeScalingConfig.factor);

  if (scalingTypeRaw == null && factor == null) {
    if (strictMissingTypeAndFactor) {
      throw new Error(
        `${sourceLabel} includes rope_scaling but is missing type/rope_type and factor. ` +
        'Provide a scaling type or factor to build manifest inference.'
      );
    }
    return {
      ropeScalingType: null,
      ropeScalingFactor: 1.0,
      yarnBetaFast: null,
      yarnBetaSlow: null,
      yarnOriginalMaxPos: null,
    };
  }

  let ropeScalingType = scalingType;
  let ropeScalingFactor = 1.0;
  let yarnBetaFast = null;
  let yarnBetaSlow = null;
  let yarnOriginalMaxPos = null;

  if (ropeScalingType == null) {
    if (factor != null && factor > 0 && factor !== 1.0) {
      ropeScalingType = 'linear';
      ropeScalingFactor = factor;
    }
  } else if (factor != null && factor > 0) {
    ropeScalingFactor = factor;
  }

  if (ropeScalingType === 'yarn') {
    const betaFast = asFiniteNumber(ropeScalingConfig.beta_fast);
    const betaSlow = asFiniteNumber(ropeScalingConfig.beta_slow);
    const origMaxPos = asFiniteNumber(ropeScalingConfig.original_max_position_embeddings);
    if (betaFast == null || betaSlow == null || origMaxPos == null) {
      throw new Error(
        'YARN scaling detected but required params missing in HF config. ' +
        'YARN requires beta_fast, beta_slow, and original_max_position_embeddings. ' +
        `Got: beta_fast=${betaFast}, beta_slow=${betaSlow}, original_max_position_embeddings=${origMaxPos}`
      );
    }
    yarnBetaFast = betaFast;
    yarnBetaSlow = betaSlow;
    yarnOriginalMaxPos = origMaxPos;
  }

  return {
    ropeScalingType,
    ropeScalingFactor,
    yarnBetaFast,
    yarnBetaSlow,
    yarnOriginalMaxPos,
  };
}

function hasScalingDirective(ropeScalingConfig) {
  if (!ropeScalingConfig) return false;
  return ropeScalingConfig.type != null
    || ropeScalingConfig.rope_type != null
    || ropeScalingConfig.factor != null
    || ropeScalingConfig.beta_fast != null
    || ropeScalingConfig.beta_slow != null
    || ropeScalingConfig.original_max_position_embeddings != null;
}

function hasMeaningfulScalingConfig(resolvedScaling) {
  if (!resolvedScaling) return false;
  return resolvedScaling.ropeScalingType != null
    || resolvedScaling.ropeScalingFactor !== 1.0
    || resolvedScaling.yarnBetaFast != null
    || resolvedScaling.yarnBetaSlow != null
    || resolvedScaling.yarnOriginalMaxPos != null;
}

function isSameScalingConfig(left, right) {
  return left.ropeScalingType === right.ropeScalingType
    && left.ropeScalingFactor === right.ropeScalingFactor
    && left.yarnBetaFast === right.yarnBetaFast
    && left.yarnBetaSlow === right.yarnBetaSlow
    && left.yarnOriginalMaxPos === right.yarnOriginalMaxPos;
}

function failOnConflictingScaling(sourceLabel, canonicalScaling, candidateScaling) {
  if (!hasMeaningfulScalingConfig(candidateScaling)) {
    return;
  }
  if (isSameScalingConfig(canonicalScaling, candidateScaling)) {
    return;
  }
  throw new Error(
    `${sourceLabel} scaling conflicts with top-level rope_scaling. ` +
    'Doppler treats rope_scaling as highest precedence and cannot safely auto-resolve this mismatch. ' +
    'Remove one source or align both scaling configs.'
  );
}

export function buildRoPEConfig(presetInference, config) {
  const ropeScaling = asObject(config.rope_scaling);
  const ropeParameters = asObject(config.rope_parameters);
  const flatRoPEParameters = (
    ropeParameters
      && !asObject(ropeParameters.full_attention)
      && !asObject(ropeParameters.sliding_attention)
  )
    ? ropeParameters
    : null;
  const fullAttentionRoPE = asObject(ropeParameters?.full_attention);
  const slidingAttentionRoPE = asObject(ropeParameters?.sliding_attention);
  const presetRoPE = presetInference.rope ?? {};
  const presetAttn = presetInference.attention;

  let globalScaling = {
    ropeScalingType: presetRoPE.ropeScalingType
      ?? presetAttn?.ropeScalingType  // Deprecated location
      ?? null,
    ropeScalingFactor: presetRoPE.ropeScalingFactor
      ?? presetAttn?.ropeScalingFactor  // Deprecated location
      ?? 1.0,
    yarnBetaFast: presetRoPE.yarnBetaFast ?? null,
    yarnBetaSlow: presetRoPE.yarnBetaSlow ?? null,
    yarnOriginalMaxPos: presetRoPE.yarnOriginalMaxPos ?? null,
  };

  if (ropeScaling) {
    // HF rope_scaling is source of truth when present.
    globalScaling = resolveScalingConfig(ropeScaling, {
      strictMissingTypeAndFactor: true,
      sourceLabel: 'HF config',
    });
    if (slidingAttentionRoPE && hasScalingDirective(slidingAttentionRoPE)) {
      failOnConflictingScaling(
        'HF config rope_parameters.sliding_attention',
        globalScaling,
        resolveScalingConfig(slidingAttentionRoPE, {
          strictMissingTypeAndFactor: false,
          sourceLabel: 'HF config rope_parameters.sliding_attention',
        })
      );
    }
  } else if (fullAttentionRoPE) {
    // Gemma 3 style rope_parameters uses per-layer-type settings.
    globalScaling = resolveScalingConfig(fullAttentionRoPE, {
      strictMissingTypeAndFactor: false,
      sourceLabel: 'HF config rope_parameters.full_attention',
    });
  } else if (flatRoPEParameters) {
    globalScaling = resolveScalingConfig(flatRoPEParameters, {
      strictMissingTypeAndFactor: false,
      sourceLabel: 'HF config rope_parameters',
    });
  }

  const hasPresetLocalScaling = presetRoPE.ropeLocalScalingType !== undefined
    || presetRoPE.ropeLocalScalingFactor !== undefined
    || presetRoPE.ropeLocalYarnBetaFast !== undefined
    || presetRoPE.ropeLocalYarnBetaSlow !== undefined
    || presetRoPE.ropeLocalYarnOriginalMaxPos !== undefined;
  let localScaling = hasPresetLocalScaling
    ? {
        ropeScalingType: presetRoPE.ropeLocalScalingType ?? globalScaling.ropeScalingType,
        ropeScalingFactor: presetRoPE.ropeLocalScalingFactor ?? globalScaling.ropeScalingFactor,
        yarnBetaFast: presetRoPE.ropeLocalYarnBetaFast ?? globalScaling.yarnBetaFast,
        yarnBetaSlow: presetRoPE.ropeLocalYarnBetaSlow ?? globalScaling.yarnBetaSlow,
        yarnOriginalMaxPos: presetRoPE.ropeLocalYarnOriginalMaxPos ?? globalScaling.yarnOriginalMaxPos,
      }
    : { ...globalScaling };
  if (ropeScaling) {
    localScaling = { ...globalScaling };
  } else if (slidingAttentionRoPE) {
    localScaling = resolveScalingConfig(slidingAttentionRoPE, {
      strictMissingTypeAndFactor: false,
      sourceLabel: 'HF config rope_parameters.sliding_attention',
    });
  }

  // HF config is source of truth for ropeTheta when provided:
  // prefer rope_parameters.full_attention.rope_theta, then rope_theta.
  const ropeTheta = asFiniteNumber(fullAttentionRoPE?.rope_theta)
    ?? asFiniteNumber(flatRoPEParameters?.rope_theta)
    ?? asFiniteNumber(config.rope_theta)
    ?? presetInference.rope?.ropeTheta
    ?? 10000;

  // For Gemma 3, local sliding attention theta comes from rope_parameters.sliding_attention.
  const ropeLocalTheta = asFiniteNumber(slidingAttentionRoPE?.rope_theta)
    ?? presetInference.rope?.ropeLocalTheta
    ?? null;

  const mropeInterleaved = asBoolean(flatRoPEParameters?.mrope_interleaved)
    ?? presetInference.rope?.mropeInterleaved
    ?? false;
  const mropeSection = asNumberArray(flatRoPEParameters?.mrope_section)
    ?? presetInference.rope?.mropeSection
    ?? null;
  const partialRotaryFactor = asFiniteNumber(flatRoPEParameters?.partial_rotary_factor)
    ?? asFiniteNumber(presetInference.rope?.partialRotaryFactor)
    ?? null;

  return {
    ropeTheta,
    ropeLocalTheta,
    mropeInterleaved,
    mropeSection,
    partialRotaryFactor,
    ropeScalingType: globalScaling.ropeScalingType,
    ropeScalingFactor: globalScaling.ropeScalingFactor,
    yarnBetaFast: globalScaling.yarnBetaFast,
    yarnBetaSlow: globalScaling.yarnBetaSlow,
    yarnOriginalMaxPos: globalScaling.yarnOriginalMaxPos,
    ropeLocalScalingType: localScaling.ropeScalingType,
    ropeLocalScalingFactor: localScaling.ropeScalingFactor,
    ropeLocalYarnBetaFast: localScaling.yarnBetaFast,
    ropeLocalYarnBetaSlow: localScaling.yarnBetaSlow,
    ropeLocalYarnOriginalMaxPos: localScaling.yarnOriginalMaxPos,
  };
}
