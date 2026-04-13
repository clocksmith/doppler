import type { RuntimeConfigSchema } from '../config/schema/index.js';
import type { PipelineStorageContext } from '../inference/pipelines/text/init.js';
import type { RuntimeModelContract } from '../inference/runtime-model.js';
import type { DirectSourceRuntimeKind } from './source-artifact-adapter.js';

export interface ResolveNodeSourceRuntimeBundleOptions {
  inputPath: string;
  modelId?: string | null;
  verifyHashes?: boolean;
  runtimeConfig?: RuntimeConfigSchema | null;
}

export interface NodeSourceRuntimeBundle {
  model: RuntimeModelContract;
  manifest: RuntimeModelContract;
  storageContext: PipelineStorageContext;
  sourceKind: DirectSourceRuntimeKind;
  sourceRoot: string;
  resolvedMemoryBudgetBytes: number | null;
}

export declare function resolveNodeSourceRuntimeBundle(
  options: ResolveNodeSourceRuntimeBundleOptions
): Promise<NodeSourceRuntimeBundle | null>;
