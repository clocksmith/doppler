export interface AttentionPrecisionContract {
  precision: {
    activationDtype?: 'f16' | 'f32';
    kvDtype?: 'f16' | 'f32';
    inputDtype?: 'f16' | 'f32';
    outputDtype?: 'f16' | 'f32';
  } | null;
  explicitInputDtype: 'f16' | 'f32' | null;
  explicitOutputDtype: 'f16' | 'f32' | null;
  explicitKvDtype: 'f16' | 'f32' | null;
  resolvedActivationDtype: 'f16' | 'f32' | null;
  resolvedOutputDtype: 'f16' | 'f32' | null;
  resolvedKvCacheDtype: 'f16' | 'f32' | null;
}

export function resolveAttentionPrecisionContract(
  config: {
    isPrefill?: boolean;
    layerIdx?: number;
    kernelPath?: Record<string, unknown> | null;
    activationDtype?: 'f16' | 'f32' | null;
    inputDtype?: 'f16' | 'f32' | null;
    outputDtype?: 'f16' | 'f32' | null;
    kvDtype?: 'f16' | 'f32' | null;
  } | null | undefined,
  state: {
    kvCache?: {
      kvDtype?: 'f16' | 'f32' | null;
    } | null;
  } | null | undefined
): AttentionPrecisionContract;

export function isAttentionKvDtypeExplicit(
  contract: AttentionPrecisionContract | null | undefined,
  targetDtype: 'f16' | 'f32'
): boolean;
