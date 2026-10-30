export interface TensorSource {
  sourceType?: string;
  name?: string | null;
  size: number;
  file?: Blob | File;
  readRange: (offset: number, length: number) => Promise<ArrayBuffer>;
  readAll?: () => Promise<ArrayBuffer>;
  close?: () => Promise<void>;
  getAuxFiles?: () => Promise<Record<string, unknown>> | Record<string, unknown>;
}

export interface GGUFParserAdapter {
  file: File | TensorSource;
  parseGGUFHeaderFromSource: (source: TensorSource) => Promise<Record<string, unknown>>;
  normalizeTensorSource: (file: File | TensorSource) => TensorSource;
  onProgress?: (update: { stage?: string; message?: string }) => void;
  signal?: AbortSignal | null;
}

export interface ParsedGGUFModel {
  format: 'gguf';
  modelType?: 'transformer';
  normalizedArchitecture: Record<string, unknown> | null;
  excludedTensors: Array<{name: string; reason: string}>;
  tokenizerJson: Record<string, unknown> | null;
  tokenizerConfig: Record<string, unknown> | null;
  tensors: Array<Record<string, unknown>>;
  config: Record<string, unknown>;
  architecture: string;
  quantization: string;
  tensorDataOffset: number;
  file: File;
  source: TensorSource;
}

export declare function parseGGUFModel(adapter: GGUFParserAdapter): Promise<ParsedGGUFModel>;
