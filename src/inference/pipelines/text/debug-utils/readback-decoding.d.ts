export declare function f16ToF32(h: number): number;

export declare function decodeReadback(
  buffer: ArrayBuffer,
  dtype: 'f16' | 'f32' | 'bf16'
): Float32Array;
