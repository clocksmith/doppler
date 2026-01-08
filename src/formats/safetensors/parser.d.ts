/**
 * SafeTensors Format Parser
 * Parses HuggingFace safetensors files for tensor metadata and data.
 */

import type { SafetensorsTensor as CoreSafetensorsTensor } from './types.js';

export { DTYPE_SIZE, DTYPE_MAP } from './types.js';
export type { SafetensorsHeader, SafetensorsHeaderInfo } from './types.js';

export type SafetensorsTensor = CoreSafetensorsTensor & {
  filePath?: string;
  shardPath?: string;
};

export interface ParsedHeader {
  headerSize: number;
  dataOffset: number;
  metadata: Record<string, string>;
  tensors: SafetensorsTensor[];
}

export interface ShardInfo {
  file: string;
  path: string;
  size: number;
  tensorCount: number;
}

export interface ParsedSafetensorsFile {
  dataOffset: number;
  metadata: Record<string, string>;
  tensors: SafetensorsTensor[];
  filePath: string;
  fileSize: number;
  config?: Record<string, unknown>;
  tokenizerConfig?: Record<string, unknown>;
  tokenizerJson?: Record<string, unknown>;
}

export interface ParsedSafetensorsIndex {
  indexPath: string;
  modelDir: string;
  metadata: Record<string, unknown>;
  config: Record<string, unknown>;
  shards: ShardInfo[];
  tensors: SafetensorsTensor[];
  shardParsed: Map<string, ParsedSafetensorsFile>;
  tokenizerConfig?: Record<string, unknown>;
  tokenizerJson?: Record<string, unknown>;
}

export interface ModelFormatInfo {
  sharded: boolean;
  indexPath?: string;
  singlePath?: string;
  files?: string[];
}

export declare function parseSafetensorsHeader(buffer: ArrayBuffer): ParsedHeader;

export declare function parseSafetensorsFile(filePath: string): Promise<ParsedSafetensorsFile>;

export declare function parseSafetensorsIndex(indexPath: string): Promise<ParsedSafetensorsIndex>;

export declare function loadModelConfig(modelDir: string): Promise<Record<string, unknown> | null>;

export declare function loadTokenizerConfig(modelDir: string): Promise<Record<string, unknown> | null>;

export declare function loadTokenizerJson(modelDir: string): Promise<Record<string, unknown> | null>;

export declare function detectModelFormat(modelDir: string): Promise<ModelFormatInfo>;

export declare function parseSafetensors(
  pathOrDir: string
): Promise<ParsedSafetensorsFile | ParsedSafetensorsIndex>;

export declare function getTensor(
  parsed: ParsedSafetensorsFile | ParsedSafetensorsIndex,
  name: string
): SafetensorsTensor | null;

export declare function getTensors(
  parsed: ParsedSafetensorsFile | ParsedSafetensorsIndex,
  pattern: RegExp
): SafetensorsTensor[];

export declare function readTensorData(
  tensor: SafetensorsTensor,
  buffer?: ArrayBuffer
): Promise<ArrayBuffer>;

export declare function groupTensorsByLayer(
  parsed: ParsedSafetensorsFile | ParsedSafetensorsIndex
): Map<number, SafetensorsTensor[]>;

export declare function calculateTotalSize(parsed: ParsedSafetensorsFile | ParsedSafetensorsIndex): number;
