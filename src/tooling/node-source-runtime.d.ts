import type { RDRRManifest } from '../formats/rdrr/index.js';
import type { RuntimeConfigSchema } from '../config/schema/index.js';
import type { PipelineStorageContext } from '../inference/pipelines/text/init.js';

export interface ResolveNodeSourceRuntimeBundleOptions {
  inputPath: string;
  modelId?: string | null;
  verifyHashes?: boolean;
  runtimeConfig?: RuntimeConfigSchema | null;
}

export interface NodeSourceRuntimeBundle {
  manifest: RDRRManifest;
  storageContext: PipelineStorageContext;
  sourceKind: 'safetensors' | 'gguf';
  sourceRoot: string;
  resolvedMemoryBudgetBytes: number | null;
}

export declare function resolveNodeSourceRuntimeBundle(
  options: ResolveNodeSourceRuntimeBundleOptions
): Promise<NodeSourceRuntimeBundle | null>;
