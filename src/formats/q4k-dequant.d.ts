export declare function dequantizeQ4KM(
  quantized: Uint8Array,
  numBlocks: number,
  shape: number[]
): Float32Array;

export declare function dequantizeQ4KMRowWise(
  quantized: Uint8Array,
  shape: number[]
): Float32Array;
