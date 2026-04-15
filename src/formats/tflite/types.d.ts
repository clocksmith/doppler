export declare const TFLITE_FILE_IDENTIFIER: 'TFL3';

export declare const TFLITE_TENSOR_TYPE: {
  readonly FLOAT32: 0;
  readonly FLOAT16: 1;
  readonly INT32: 2;
  readonly UINT8: 3;
  readonly INT64: 4;
  readonly STRING: 5;
  readonly BOOL: 6;
  readonly INT16: 7;
  readonly COMPLEX64: 8;
  readonly INT8: 9;
  readonly FLOAT64: 10;
  readonly COMPLEX128: 11;
  readonly UINT64: 12;
  readonly RESOURCE: 13;
  readonly VARIANT: 14;
  readonly UINT32: 15;
  readonly UINT16: 16;
  readonly INT4: 17;
  readonly BFLOAT16: 18;
};

export type TFLiteTensorTypeId = (typeof TFLITE_TENSOR_TYPE)[keyof typeof TFLITE_TENSOR_TYPE];

export declare const TFLITE_TENSOR_TYPE_NAME: Record<number, string>;

export declare const TFLITE_TENSOR_DTYPE_MAP: Record<number, 'F32' | 'F16' | 'BF16'>;
export declare const TFLITE_TENSOR_SOURCE_DTYPE_MAP: Record<number, 'F32' | 'F16' | 'BF16' | 'INT8' | 'UINT8' | 'INT4'>;

export declare const TFLITE_DTYPE_SIZE: Record<'F32' | 'F16' | 'BF16', number>;

export interface TFLiteSourceTransform {
  kind: 'affine_dequant';
  scheme: 'per_tensor_affine';
  sourceDtype: 'INT8' | 'UINT8' | 'INT4';
  targetDtype: 'F16';
  scale: number;
  zeroPoint: number;
}

export interface TFLiteTensor {
  name: string;
  shape: number[];
  dtype: string;
  dtypeId: number;
  sourceDtype: string;
  offset: number;
  size: number;
  buffer: number;
  subgraphIndex: number;
  isVariable: boolean;
  sourceTransform?: TFLiteSourceTransform;
}

export interface TFLiteMetadataEntry {
  name: string;
  buffer: number;
  offset: number;
  size: number;
}

export interface ParsedTFLite {
  schemaVersion: number;
  description: string | null;
  subgraphCount: number;
  mainSubgraphName: string | null;
  tensors: TFLiteTensor[];
  metadataEntries: TFLiteMetadataEntry[];
  sourceQuantization: 'F32' | 'F16' | 'BF16' | null;
}

export interface TFLiteSource {
  name?: string | null;
  size: number;
  readRange: (offset: number, length: number) => Promise<ArrayBuffer | Uint8Array>;
}

export interface ParseTFLiteOptions {
  allowPackedQuantization?: boolean;
}

export declare function parseTFLite(
  buffer: ArrayBuffer | Uint8Array,
  options?: ParseTFLiteOptions
): Promise<ParsedTFLite>;

export declare function parseTFLiteFromSource(
  source: TFLiteSource,
  options?: ParseTFLiteOptions
): Promise<ParsedTFLite>;
