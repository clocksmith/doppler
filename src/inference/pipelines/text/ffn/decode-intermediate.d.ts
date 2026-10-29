export interface FfnIntermediateContext {
  phase?: string;
  activationPrecision?: 'f32' | 'two-f32' | null;
  trainingCapture?: unknown;
  operatorDiagnostics?: unknown;
  debugProbes?: readonly unknown[] | null;
  decodeBuffers?: { getFFNIntermediateBuffer(): GPUBuffer | null } | null;
}
/** Borrows the session-owned scratch vector. GPU command order places the next
 * producer after the preceding down projection. The caller must not release it.
 * Retained observation/training consumers keep independent ordinary outputs.
 */
export function borrowDecodeFfnIntermediate(
  context: FfnIntermediateContext, numTokens: number, intermediateSize: number, dtype: string
): GPUBuffer | null;
