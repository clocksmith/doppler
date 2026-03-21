import type { ManifestEmbeddingPostprocessorSchema, TensorRole } from '../../config/schema/index.js';

export interface ParsedTransformerTensor {
  name: string;
  shape?: number[];
  dtype?: string;
  size?: number;
  offset?: number;
  sourcePath?: string;
  role?: TensorRole;
  group?: string | null;
}

export interface TransformerParserAdapter {
  readJson: (suffix: string, label?: string) => Promise<Record<string, unknown>>;
  fileExists: (suffix: string) => Promise<boolean>;
  loadSingleSafetensors: (suffix: string) => Promise<ParsedTransformerTensor[]>;
  loadShardedSafetensors: (indexJson: Record<string, unknown>) => Promise<ParsedTransformerTensor[]>;
}

export interface ParsedTransformerModel {
  config: Record<string, unknown>;
  generationConfig: Record<string, unknown> | null;
  tensors: ParsedTransformerTensor[];
  architectureHint: string;
  embeddingPostprocessor: ManifestEmbeddingPostprocessorSchema | null;
}

export declare function parseTransformerModel(
  adapter: TransformerParserAdapter
): Promise<ParsedTransformerModel>;
