import type { HashAlgorithm } from '../../config/schema/index.js';
import type { HotSwapConfigSchema } from '../../config/schema/hotswap.schema.js';

export interface HotSwapArtifact {
  path: string;
  hash: string;
  hashAlgorithm?: HashAlgorithm;
}

export interface HotSwapManifest {
  bundleId: string;
  version: string;
  artifacts: HotSwapArtifact[];
  signerId?: string;
  signature?: string;
  createdAt?: string;
  metadata?: Record<string, string>;
}

export interface HotSwapVerificationResult {
  ok: boolean;
  reason: string;
  signerId?: string;
}

export interface HotSwapVerificationContext {
  source?: {
    kind?: 'local' | 'remote' | string | null;
    isLocal?: boolean | null;
    url?: string | null;
  } | null;
}

export declare function fetchHotSwapManifest(url: string): Promise<HotSwapManifest>;

export declare function verifyHotSwapManifest(
  manifest: HotSwapManifest,
  policy: HotSwapConfigSchema,
  context?: HotSwapVerificationContext
): Promise<HotSwapVerificationResult>;

export declare function serializeHotSwapManifest(manifest: HotSwapManifest): string;
