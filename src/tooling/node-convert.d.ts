export interface NodeConvertProgress {
  stage: string | null;
  current: number | null;
  total: number | null;
  message: string | null;
}

export interface ConvertSafetensorsDirectoryOptions {
  inputDir: string;
  outputDir: string;
  modelId?: string | null;
  onProgress?: (progress: NodeConvertProgress) => void;
}

export interface ConvertSafetensorsDirectoryResult {
  manifest: Record<string, unknown>;
  shardCount: number;
  tensorCount: number;
  presetId: string;
  modelType: string;
  outputDir: string;
}

export declare function convertSafetensorsDirectory(
  options: ConvertSafetensorsDirectoryOptions
): Promise<ConvertSafetensorsDirectoryResult>;
