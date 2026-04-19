export declare const TFLITE_FILE_IDENTIFIER: 'TFL3';

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
