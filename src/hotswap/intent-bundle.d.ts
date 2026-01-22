import type { Manifest } from '../config/schema/manifest.schema.js';

export interface IntentBundleVerifyContext {
  manifest?: Manifest | null;
  kernelRegistryVersion?: string | null;
  enforceDeterministicOutput?: boolean;
}

export interface IntentBundleVerificationResult {
  ok: boolean;
  reason: string;
  reasons?: string[];
}

export interface IntentBundleDriftResult {
  ok: boolean;
  drift: number;
  matchRatio: number;
  reason: string;
}

export declare function fetchIntentBundle(url: string): Promise<Record<string, unknown>>;
export declare function computeManifestHash(manifest: Manifest): Promise<string>;
export declare function getKernelRegistryVersion(): Promise<string | null>;
export declare function verifyIntentBundle(
  bundle: Record<string, unknown>,
  context: IntentBundleVerifyContext
): Promise<IntentBundleVerificationResult>;
export declare function compareTopK(
  expectedTopK: number[],
  actualTopK: number[]
): { matchRatio: number; drift: number };
export declare function enforceLogitDrift(
  expectedTopK: number[],
  actualTopK: number[],
  maxDriftThreshold: number | null | undefined
): IntentBundleDriftResult;
