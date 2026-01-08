/**
 * Build RoPE configuration from preset and HF config.
 *
 * HF rope_scaling is treated as source of truth when present.
 */
export function buildRoPEConfig(presetInference, config) {
  // Extract rope_scaling object from HF config
  const ropeScaling = config.rope_scaling;

  // HF rope_scaling always takes precedence over preset (per-model config > family defaults)
  let ropeScalingType = null;
  let ropeScalingFactor = 1.0;

  // YARN params (only populated for YARN scaling)
  let yarnBetaFast = null;
  let yarnBetaSlow = null;
  let yarnOriginalMaxPos = null;

  if (ropeScaling && typeof ropeScaling === 'object') {
    // HF config is source of truth for rope_scaling
    const scalingType = ropeScaling.type ?? ropeScaling.rope_type;
    const factor = ropeScaling.factor;
    if (scalingType == null) {
      if (factor != null && factor > 0) {
        // Infer linear scaling when factor is present but type is missing.
        ropeScalingType = 'linear';
        ropeScalingFactor = factor;
      } else {
        throw new Error(
          'HF config includes rope_scaling but is missing type/rope_type and factor. ' +
          'Provide a scaling type or factor to build manifest inference.'
        );
      }
    } else {
      ropeScalingType = scalingType;
      if (factor != null && factor > 0) {
        ropeScalingFactor = factor;
      }
    }

    // Extract YARN-specific params (ALL required when type='yarn' - fail fast)
    if (ropeScalingType === 'yarn') {
      const betaFast = ropeScaling.beta_fast;
      const betaSlow = ropeScaling.beta_slow;
      const origMaxPos = ropeScaling.original_max_position_embeddings;
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
  } else {
    // No HF rope_scaling - fall back to preset (check both canonical and deprecated locations)
    const presetAttn = presetInference.attention;
    ropeScalingType = presetInference.rope?.ropeScalingType
      ?? presetAttn?.ropeScalingType  // Deprecated location
      ?? null;
    ropeScalingFactor = presetInference.rope?.ropeScalingFactor
      ?? presetAttn?.ropeScalingFactor  // Deprecated location
      ?? 1.0;
  }

  // HF config is source of truth for ropeTheta, preset is fallback
  const ropeTheta = config.rope_theta
    ?? presetInference.rope?.ropeTheta
    ?? 10000;

  // ropeLocalTheta is model-family specific (e.g., Gemma 3), comes from preset
  const ropeLocalTheta = presetInference.rope?.ropeLocalTheta ?? null;

  return {
    ropeTheta,
    ropeLocalTheta,
    ropeScalingType,
    ropeScalingFactor,
    yarnBetaFast,
    yarnBetaSlow,
    yarnOriginalMaxPos,
  };
}
