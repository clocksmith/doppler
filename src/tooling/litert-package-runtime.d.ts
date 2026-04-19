import type { ParsedSourceArtifact } from './source-artifact-adapter.js';
import type { DirectSourcePackageProfile } from './source-package-profiles.js';
import type { LiteRTSource } from '../formats/litert/types.js';

export declare const LITERT_PACKAGE_SOURCE_KIND_TASK: 'litert-task';
export declare const LITERT_PACKAGE_SOURCE_KIND_LITERTLM: 'litertlm';

export declare function inferLiteRTRowwiseLayout(
  rawTensor: { dtypeId?: number | null; size?: number | null; name?: string | null },
  expectedShape: [number, number],
  tensorName?: string
): {
  sourceDtype: 'INT8' | 'UINT8' | 'INT4' | 'INT2';
  storageEncoding: 'signed' | 'offset_binary';
};

export declare function resolveGemma4AttentionHeadDim(
  runtimeProfile: {
    architecture?: { headDim?: number | null; globalHeadDim?: number | null } | null;
    manifestInference?: { layerPattern?: { type?: string | null; period?: number | null; offset?: number | null } | null } | null;
  } | null | undefined,
  layerIndex: number
): number;

export interface LiteRTPackageVirtualFile {
  path: string;
  offset: number;
  size: number;
  kind: string;
  externalPath?: string | null;
}

export declare function resolveLiteRTPackageParsedArtifact(options: {
  sourceKind: 'litert-task' | 'litertlm';
  source: LiteRTSource;
  sourcePathForModelId?: string | null;
}): Promise<{
  parsedArtifact: ParsedSourceArtifact;
  virtualFiles: LiteRTPackageVirtualFile[];
  packageProfile: DirectSourcePackageProfile;
}>;

export declare function appendLiteRTPackageVirtualFiles(
  virtualFiles: LiteRTPackageVirtualFile[] | null | undefined,
  entries?: LiteRTPackageVirtualFile[] | null
): LiteRTPackageVirtualFile[];
