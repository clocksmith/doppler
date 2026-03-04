import type { ExtensionBridgeClient } from '../../bridge/extension-client.js';
import type { RDRRManifest } from '../../formats/rdrr/index.js';
import type { PipelineStorageContext } from '../../inference/pipelines/text/init.js';

export interface ResolveBridgeSourceRuntimeBundleOptions {
  bridgeClient: ExtensionBridgeClient;
  localPath: string;
  modelId?: string | null;
  onProgress?: (info: { stage: string; message: string }) => void;
}

export interface BridgeSourceRuntimeBundle {
  manifest: RDRRManifest;
  storageContext: PipelineStorageContext;
  sourceKind: 'safetensors' | 'gguf';
  sourceRoot: string;
}

export declare function resolveBridgeSourceRuntimeBundle(
  options: ResolveBridgeSourceRuntimeBundleOptions
): Promise<BridgeSourceRuntimeBundle | null>;

