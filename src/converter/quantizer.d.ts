/**
 * Tensor Quantization for .rdrr Format
 * Implements Q4_K_M quantization (4-bit with k-means clustering).
 *
 * @module converter/quantizer
 */

/** Q4K block size in elements (256 weights per block) */
export declare const QK_K: number;

/** K-means scale size for Q4K */
export declare const K_SCALE_SIZE: number;

/** Q4K block size in bytes (144 bytes per 256 weights) */
export declare const QK4_K_BLOCK_SIZE: number;

export interface QuantizeResult {
  quantized: Uint8Array;
  numBlocks: number;
  originalSize: number;
  quantizedSize: number;
  compressionRatio: number;
}

export interface QuantizationError {
  mse: number;
  maxError: number;
  snr: number;
}

export declare function float32ToFloat16(value: number): number;

export declare function float16ToFloat32(h: number): number;

/**
 * Flat Q4K quantization (original behavior).
 * Packs all elements sequentially - blocks may cross row boundaries.
 */
export declare function quantizeToQ4KM(data: Float32Array, shape: number[]): QuantizeResult;

/**
 * Row-wise Q4K quantization for 2D weight matrices.
 * Each row is padded to 256-element boundary before quantization.
 */
export declare function quantizeToQ4KMRowWise(data: Float32Array, shape: [number, number]): QuantizeResult;

/**
 * Transpose a 2D F32 matrix.
 */
export declare function transposeF32(data: Float32Array, shape: [number, number]): Float32Array;

/**
 * Column-wise Q4K quantization for 2D weight matrices.
 * Transposes W[out, K] → W^T[K, out], then applies row-wise quantization.
 */
export declare function quantizeToQ4KMColumnWise(
  data: Float32Array,
  shape: [number, number]
): QuantizeResult & { transposedShape: [number, number] };

export type Q4KLayout = 'flat' | 'row_wise' | 'column_wise';

/**
 * Get expected Q4K size for different layouts.
 */
export declare function getQ4KSize(shape: number[], layout?: Q4KLayout): number;

export declare function dequantizeQ4KM(
  quantized: Uint8Array,
  numBlocks: number,
  shape: number[]
): Float32Array;

export declare function calculateQuantizationError(
  original: Float32Array,
  reconstructed: Float32Array
): QuantizationError;

export declare function quantizeF16ToQ4KM(f16Data: Uint16Array, shape: number[]): QuantizeResult;

export interface QuantizeOptions {
  /** Also quantize embedding tables (default: false) */
  quantizeEmbeddings?: boolean;
  /** Modules to skip (from HF config) */
  modulesToNotConvert?: string[] | null;
}

export declare function shouldQuantize(
  tensorName: string,
  shape: number[],
  options?: QuantizeOptions
): boolean;

export declare function getQuantizedSize(shape: number[]): number;
