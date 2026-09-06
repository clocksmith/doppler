import type { DopplerCapsuleV2 } from '../config/capsule-v2.js';
import type { ForgeEvaluationInput } from './forge-candidate-evaluation.js';

export const FORGE_PIPELINE_VERSION: '2.0.0';

export interface ForgeStageResult extends Record<string, unknown> {
  stage: string;
  ok: true;
}

export declare function stageInspect(input: Record<string, unknown>): Promise<ForgeStageResult & { data: Record<string, unknown> }>;
export declare function stageNormalize(input: Record<string, unknown>): ForgeStageResult;
export declare function stageAnalyze(input: Record<string, unknown>): ForgeStageResult;
export declare function stageLower(input: Record<string, unknown>): ForgeStageResult;
export declare function stageSpecialize(input: Record<string, unknown>): ForgeStageResult;
export declare function stageSearch(input: Record<string, unknown>, evaluation?: ForgeEvaluationInput): ForgeStageResult;
export declare function stageVerify(input: Record<string, unknown>): ForgeStageResult;
export declare function stageQualify(input: Record<string, unknown>): ForgeStageResult;
export declare function stagePackage(input: Record<string, unknown>): ForgeStageResult & { capsule: DopplerCapsuleV2 };
export declare function stageSign(input: ForgeStageResult & { capsule: DopplerCapsuleV2 }, signer: {
  authority: string;
  privateKeyJwk: JsonWebKey;
  publicKeyJwk: JsonWebKey;
}): Promise<ForgeStageResult & { capsule: DopplerCapsuleV2 }>;
export declare function runForgePipeline(input: Record<string, unknown>, signer: {
  authority: string;
  privateKeyJwk: JsonWebKey;
  publicKeyJwk: JsonWebKey;
}): Promise<{ capsule: DopplerCapsuleV2; searchReceipt: Record<string, unknown>; stages: Array<{ stage: string; ok: true }> }>;
