export interface QwenGGUFSource {
  config: Record<string, unknown>;
  textLayers: number;
  modelType: 'transformer';
}
export declare function normalizeQwenGGUFSource(architecture: string, metadata: Record<string, unknown>): QwenGGUFSource | null;
