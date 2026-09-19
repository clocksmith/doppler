/**
 * RoPE configuration for inference.
 * All fields required - converter must populate everything.
 * This is the canonical source for RoPE params (not architecture.ropeTheta).
 */
export interface ManifestRoPESchema {
  /** Base theta for rotary embeddings (canonical source for execution) */
  ropeTheta: number;
  /** Local theta for sliding window layers (null = same as ropeTheta) */
  ropeLocalTheta: number | null;
  /** Layer indices that bypass rotary position embeddings. */
  disabledLayers: number[];
  /** Use adjacent-pair rotary layout instead of rotate-half layout for standard RoPE. */
  ropeInterleaved: boolean;
  /** Use adjacent-pair rotary layout for mRoPE section pairing. */
  mropeInterleaved: boolean;
  /** mRoPE section sizes before the Qwen doubling step. */
  mropeSection: number[] | null;
  /** Fraction of the head dimension that participates in rotary embedding. */
  partialRotaryFactor: number | null;
  /** Fraction of the local/sliding head dimension that participates in rotary embedding. */
  ropeLocalPartialRotaryFactor: number | null;
  /** Exact source f32 inverse frequencies; null uses the declared theta formula. */
  ropeInverseFrequencies: number[] | null;
  /** Frequency exponent base dimension for full/global attention RoPE (null = use rotary dim). */
  ropeFrequencyBaseDim: number | null;
  /** Frequency exponent base dimension for local/sliding attention RoPE (null = use local rotary dim). */
  ropeLocalFrequencyBaseDim: number | null;
  /** RoPE scaling type (null = no scaling, 'linear', 'dynamic', 'yarn') */
  ropeScalingType: string | null;
  /** RoPE scaling factor (1.0 if no scaling) */
  ropeScalingFactor: number;
  /** Local RoPE scaling type for sliding window layers (null = no local scaling) */
  ropeLocalScalingType: string | null;
  /** Local RoPE scaling factor for sliding window layers (1.0 if no local scaling) */
  ropeLocalScalingFactor: number;
  /** YARN beta_fast parameter (null if not YARN scaling) */
  yarnBetaFast: number | null;
  /** YARN beta_slow parameter (null if not YARN scaling) */
  yarnBetaSlow: number | null;
  /** YARN original max position embeddings (null if not YARN scaling) */
  yarnOriginalMaxPos: number | null;
  /** LongRoPE short-context factors (null if not LongRoPE scaling) */
  longropeShortFactor: number[] | null;
  /** LongRoPE long-context factors (null if not LongRoPE scaling) */
  longropeLongFactor: number[] | null;
  /** LongRoPE original max position embeddings (null if not LongRoPE scaling) */
  longropeOriginalMaxPos: number | null;
  /** Local YARN beta_fast parameter (null if not local YARN scaling) */
  ropeLocalYarnBetaFast: number | null;
  /** Local YARN beta_slow parameter (null if not local YARN scaling) */
  ropeLocalYarnBetaSlow: number | null;
  /** Local YARN original max position embeddings (null if not local YARN scaling) */
  ropeLocalYarnOriginalMaxPos: number | null;
}
