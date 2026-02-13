export interface TransformerParserAdapter {
  readJson: (suffix: string, label?: string) => Promise<Record<string, unknown>>;
  fileExists: (suffix: string) => Promise<boolean>;
  loadSingleSafetensors: (suffix: string) => Promise<Array<Record<string, unknown>>>;
  loadShardedSafetensors: (indexJson: Record<string, unknown>) => Promise<Array<Record<string, unknown>>>;
}

export interface ParsedTransformerModel {
  config: Record<string, unknown>;
  tensors: Array<Record<string, unknown>>;
  architectureHint: string;
}

export declare function parseTransformerModel(
  adapter: TransformerParserAdapter
): Promise<ParsedTransformerModel>;
