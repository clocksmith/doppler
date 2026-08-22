import type { ModelIR } from '../config/model-ir.js';
import type { TargetPlan, TargetPlanKernelModule } from '../config/target-plan.js';

export const PACK_V2_SCHEMA_ID = 'doppler.pack/v2';
export const PACK_V2_SCHEMA_VERSION = 2;

export interface PackV2Artifact {
  role: 'manifest' | 'weight-shard' | 'tokenizer' | 'conversion-config' | 'reference-report' | 'model-ir' | 'target-plan' | 'signature';
  path: string;
  hash: `sha256:${string}`;
  sizeBytes: number;
}

export interface PackV2Signature {
  authority: string;
  algorithm: string;
  publicKeyDigest: string;
  signatureHex: string;
  signedDigest: string;
}

export interface DopplerPackV2 {
  schema: 'doppler.pack/v2';
  schemaVersion: 2;
  packId: string;
  modelId: string;
  createdAtUtc: string;
  modelIR: ModelIR;
  targetPlans: TargetPlan[];
  wgslModules: TargetPlanKernelModule[];
  artifacts: PackV2Artifact[];
  signature: PackV2Signature | null;
}

export declare function validatePackV2(pack: unknown): { ok: boolean; errors: string[] };
export declare function hashPackV2(pack: unknown): `sha256:${string}`;
export declare function buildPackV2(params: Partial<DopplerPackV2>): DopplerPackV2;
export declare function writePackV2(outputPath: string, pack: DopplerPackV2): Promise<{ ok: boolean; outputPath: string; packHash: string }>;
export declare function loadPackV2(packPath: string): Promise<DopplerPackV2>;
