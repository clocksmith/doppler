/**
 * Tensor and Data Types
 */

/** Data type enumeration */
export type DType =
  | 'f32'
  | 'f16'
  | 'bf16'
  | 'i32'
  | 'u32'
  | 'i16'
  | 'u16'
  | 'i8'
  | 'u8';

/** Quantized data types */
export type QType =
  | 'q8_0'
  | 'q4_0'
  | 'q4_1'
  | 'q4_k'
  | 'q5_k'
  | 'q6_k'
  | 'q8_k';

/** All supported data types */
export type DataType = DType | QType;

/** Data type information */
export interface DTypeInfo {
  name: DataType;
  bytesPerElement: number;
  alignment: number;
  isQuantized: boolean;
  blockSize?: number;
  bitsPerWeight?: number;
}

/** Tensor shape */
export type TensorShape = number[];

/** Tensor metadata */
export interface TensorMetadata {
  name: string;
  shape: TensorShape;
  dtype: DataType;
  byteOffset: number;
  byteLength: number;
  shardIndex: number;
}

/** Tensor request for loading */
export interface TensorRequest {
  name: string;
  aliases?: string[];
  required: boolean;
  expectedDtype?: DataType;
  expectedShape?: TensorShape;
}

/** Loaded tensor result */
export interface LoadedTensor {
  name: string;
  shape: TensorShape;
  dtype: DataType;
  buffer: GPUBuffer;
  byteLength: number;
}

/** Tensor span within a shard */
export interface TensorSpan {
  tensorName: string;
  shardIndex: number;
  byteOffset: number;
  byteLength: number;
}

/** Tensor addressing table entry */
export interface TensorAddress {
  name: string;
  shardIndex: number;
  offset: number;
  length: number;
  dtype: DataType;
  shape: TensorShape;
}
