export declare function emitObservedLogits(
  onLogits: ((logits: Float32Array, context: { tokenId: number; inputTokenCount: number | null }) => void) | null | undefined,
  logits: Float32Array,
  tokenId: number,
  currentIds: number[]
): boolean;

export declare function captureObservedFusedDecodeLogits(
  state: Record<string, unknown>,
  opts: Record<string, unknown>,
  logitsBuffer: GPUBuffer,
  vocabSize: number,
  logitsDtype: string,
  tokenId: number,
  currentIds: number[]
): Promise<boolean>;
