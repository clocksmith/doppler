import type { RDRRManifest } from '../storage/rdrr-format.js';
import type { PipelineStorageContext } from '../inference/pipelines/text/init.js';

export interface ResolveNodeSourceRuntimeBundleOptions {
  inputPath: string;
  modelId?: string | null;
}

export interface NodeSourceRuntimeBundle {
  manifest: RDRRManifest;
  storageContext: PipelineStorageContext;
  sourceKind: 'safetensors' | 'gguf';
  sourceRoot: string;
}

export declare function resolveNodeSourceRuntimeBundle(
  options: ResolveNodeSourceRuntimeBundleOptions
): Promise<NodeSourceRuntimeBundle | null>;

