import type { PipelineStorageContext } from '../../inference/pipelines/text/init.js';
import type { RuntimeModelContract } from '../../inference/runtime-model.js';
import type { DirectSourceRuntimeKind } from '../../tooling/source-artifact-adapter.js';
import type { ExtensionBridgeClient } from './types.js';

export interface ResolveBridgeSourceRuntimeBundleOptions {
  bridgeClient: ExtensionBridgeClient;
  localPath: string;
  modelId?: string | null;
  model?: RuntimeModelContract | null;
  manifest?: RuntimeModelContract | null;
  onProgress?: (info: { stage: string; message: string }) => void;
  verifyHashes?: boolean;
}

export interface BridgeSourceRuntimeBundle {
  model: RuntimeModelContract;
  manifest: RuntimeModelContract;
  storageContext: PipelineStorageContext;
  sourceKind: DirectSourceRuntimeKind | 'rdrr';
  sourceRoot: string;
}

export declare function resolveBridgeSourceRuntimeBundle(
  options: ResolveBridgeSourceRuntimeBundleOptions
): Promise<BridgeSourceRuntimeBundle | null>;
