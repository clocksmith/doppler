import type { ExtensionBridgeClient } from '../../bridge/extension-client.js';
import type { RDRRManifest } from '../../formats/rdrr/index.js';
import type { PipelineStorageContext } from '../../inference/pipelines/text/init.js';
import type { DirectSourceRuntimeKind } from '../../tooling/source-artifact-adapter.js';

export interface ResolveBridgeSourceRuntimeBundleOptions {
  bridgeClient: ExtensionBridgeClient;
  localPath: string;
  modelId?: string | null;
  manifest?: RDRRManifest | null;
  onProgress?: (info: { stage: string; message: string }) => void;
  verifyHashes?: boolean;
}

export interface BridgeSourceRuntimeBundle {
  manifest: RDRRManifest;
  storageContext: PipelineStorageContext;
  sourceKind: DirectSourceRuntimeKind | 'rdrr';
  sourceRoot: string;
}

export declare function resolveBridgeSourceRuntimeBundle(
  options: ResolveBridgeSourceRuntimeBundleOptions
): Promise<BridgeSourceRuntimeBundle | null>;
