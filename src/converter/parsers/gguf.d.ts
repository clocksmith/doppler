import type { TensorSource } from '../../browser/tensor-source-file.js';

export interface GGUFParserAdapter {
  file: File | TensorSource;
  parseGGUFHeaderFromSource: (source: TensorSource) => Promise<Record<string, unknown>>;
  normalizeTensorSource: (file: File | TensorSource) => TensorSource;
  onProgress?: (update: { stage?: string; message?: string }) => void;
  signal?: AbortSignal | null;
}

export interface ParsedGGUFModel {
  format: 'gguf';
  tensors: Array<Record<string, unknown>>;
  config: Record<string, unknown>;
  architecture: string;
  quantization: string;
  tensorDataOffset: number;
  file: File;
  source: TensorSource;
}

export declare function parseGGUFModel(adapter: GGUFParserAdapter): Promise<ParsedGGUFModel>;
