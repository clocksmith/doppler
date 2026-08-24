export declare function getLogitsHealth(logits: Float32Array): {
  nanCount: number;
  infCount: number;
  nonZeroCount: number;
  maxAbs: number;
};

export declare function getBufferStats(
  buffer: GPUBuffer
): Promise<{ min: number; max: number; maxAbs: number; sample: number[]; nanCount: number } | null>;
