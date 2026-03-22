/**
 * Policy-driven tensor capture for operator-level differential debugging.
 *
 * Capture levels: none, metadata, slice, full.
 * Supports targeted capture and escalation around suspected divergence.
 */

export type CaptureLevel = 'none' | 'metadata' | 'slice' | 'full';

export declare const CAPTURE_LEVELS: Readonly<{
  NONE: 'none';
  METADATA: 'metadata';
  SLICE: 'slice';
  FULL: 'full';
}>;

export interface CaptureConfig {
  enabled: boolean;
  defaultLevel: CaptureLevel;
  targetLevel: CaptureLevel;
  targetOpIds: string[];
  targetOperatorClasses: string[];
  targetLayers: number[];
  sampleCount: number;
  escalation: EscalationPolicy | null;
}

export interface CaptureArtifactStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  nanCount: number;
  infCount: number;
  zeroCount: number;
  elementCount: number;
}

export interface CaptureArtifact {
  opId: string;
  level: CaptureLevel;
  timestamp: number;
  shape: number[] | null;
  dtype: string | null;
  stats?: CaptureArtifactStats | null;
  sample?: number[] | null;
  data?: number[] | null;
}

export interface EscalationPolicy {
  windowBefore: number;
  windowAfter: number;
  baseLevel: CaptureLevel;
  escalatedLevel: CaptureLevel;
  resolveForIndex(opIndex: number, suspectedDivergenceIndex: number | null): CaptureLevel;
}

export declare function resolveCapturePolicy(
  opId: string,
  config: CaptureConfig | null
): CaptureLevel;

export declare function escalateCaptureLevel(
  current: CaptureLevel,
  target: CaptureLevel
): CaptureLevel;

export declare function buildCaptureArtifact(
  opId: string,
  level: CaptureLevel,
  data: Float32Array | null,
  options?: {
    shape?: number[] | null;
    dtype?: string | null;
    sampleCount?: number;
  }
): CaptureArtifact;

export declare function createEscalationPolicy(options?: {
  windowBefore?: number;
  windowAfter?: number;
  baseLevel?: CaptureLevel;
  escalatedLevel?: CaptureLevel;
}): EscalationPolicy;

export declare function createDefaultCaptureConfig(): CaptureConfig;

export declare function validateCaptureConfig(config: unknown): boolean;
