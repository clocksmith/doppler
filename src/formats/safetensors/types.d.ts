/**
 * Shared safetensors parsing utilities (browser + tools).
 */

export type SafetensorsDtype =
  | 'F64'
  | 'F32'
  | 'F16'
  | 'BF16'
  | 'I64'
  | 'I32'
  | 'I16'
  | 'I8'
  | 'U8'
  | 'BOOL';

export type SafetensorsDType = SafetensorsDtype;

export declare const DTYPE_SIZE: Record<SafetensorsDtype, number>;

export declare const DTYPE_MAP: Record<string, string>;

export interface SafetensorsTensor {
  name: string;
  shape: number[];
  dtype: string;
  dtypeOriginal?: string;
  offset: number;
  size: number;
  elemSize?: number;
  byteSize?: number;
  shardFile?: string;
  shardPath?: string;
}

export interface SafetensorsHeaderInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface SafetensorsHeader {
  __metadata__?: Record<string, string>;
  [tensorName: string]: SafetensorsHeaderInfo | Record<string, string> | undefined;
}

export interface ParsedSafetensorsHeader {
  headerSize: number;
  dataOffset: number;
  metadata: Record<string, string>;
  tensors: SafetensorsTensor[];
}

export interface SafetensorsIndexJson {
  weight_map: Record<string, string>;
  metadata?: Record<string, unknown>;
}

export declare function parseSafetensorsIndexJsonText(text: string): SafetensorsIndexJson;

export declare function parseSafetensorsHeader(buffer: ArrayBuffer): ParsedSafetensorsHeader;

export declare function groupTensorsByLayer(
  parsed: { tensors: SafetensorsTensor[] }
): Map<number, SafetensorsTensor[]>;

export declare function calculateTotalSize(parsed: { tensors: SafetensorsTensor[] }): number;
