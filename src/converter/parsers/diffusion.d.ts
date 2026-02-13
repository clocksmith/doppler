export interface DiffusionTensor {
  name: string;
  shape: number[];
  dtype: string;
  size: number;
  offset: number;
  sourcePath?: string;
}

export interface ParsedTensorBundle {
  tensors: DiffusionTensor[];
}

export interface DiffusionParserAdapter {
  readJson: (suffix: string, label?: string) => Promise<Record<string, unknown>>;
  readText: (suffix: string, label?: string) => Promise<string>;
  readBinary: (suffix: string, label?: string) => Promise<ArrayBuffer>;
  findExistingSuffix: (suffixes: string[]) => string | null;
  parseSingleSafetensors: (
    suffix: string,
    componentId: string
  ) => Promise<ParsedTensorBundle>;
  parseShardedSafetensors: (
    indexSuffix: string,
    indexJson: Record<string, unknown>,
    componentId: string
  ) => Promise<ParsedTensorBundle>;
  onProgress?: (update: {
    stage?: string;
    message?: string;
  }) => void;
  signal?: AbortSignal | null;
}

export interface ParsedDiffusionModel {
  tensors: DiffusionTensor[];
  config: Record<string, unknown>;
  auxFiles: Array<{ name: string; data: string | ArrayBuffer }>;
  architecture: 'diffusion';
  layout: string;
}

export interface DiffusionLayout {
  id: string;
  requiredComponents: string[];
}

export declare function detectDiffusionLayout(modelIndex: Record<string, unknown>): DiffusionLayout;

export declare function parseDiffusionModel(adapter: DiffusionParserAdapter): Promise<ParsedDiffusionModel>;
