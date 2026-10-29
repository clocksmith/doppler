/** Raw manifest vision_config; only the nested records read here are typed. */
export interface RawVisionConfig {
  [key: string]: unknown;
  rope_parameters?: { rope_theta?: unknown; [key: string]: unknown } | null;
  normalization?: { mean?: unknown; std?: unknown; [key: string]: unknown } | null;
}

export type VisionConfigNormalizer = (context: VisionConfigNormalizerContext) => Record<string, unknown>;

export interface VisionConfigNormalizerContext {
  vc: RawVisionConfig;
  modelId: string;
  rawConfig: Record<string, unknown> | null | undefined;
  manifest: Record<string, unknown> | null | undefined;
  visionArchitecture: string;
  resolveRequiredVisionField(keys: string[], label: string): unknown;
  resolveRequiredPositiveInteger(keys: string[], label: string): number;
  resolveRequiredNonNegativeInteger(keys: string[], label: string): number;
  resolveRequiredPositiveNumber(value: unknown, label: string): number;
}

export declare function resolveVisionConfigNormalizer(
  visionArchitecture: string
): VisionConfigNormalizer | null;
