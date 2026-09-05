import type { DopplerPackV2 } from '../../config/pack-v2.js';
import type { TargetPlan } from '../../config/target-plan.js';
import type { DopplerRerankEvidence } from './model-session.js';

export const PACK_RERANK_RECEIPT_SCHEMA: 'doppler.pack-rerank-receipt/v1';

export interface PackRerankApplicationBinding {
  applicationId: string;
  applicationRevision: string;
  applicationRevisionDigest: `sha256:${string}`;
  workload: { id: string; digest: `sha256:${string}` };
  oracle: { id: string; digest: `sha256:${string}` };
}

export interface PackRerankRequest {
  application: PackRerankApplicationBinding;
  query: string;
  documents: string[];
  options?: { benchmark?: boolean };
}

export interface PackRerankReceipt {
  schema: 'doppler.pack-rerank-receipt/v1';
  pack: {
    packId: string;
    semanticRoot: `sha256:${string}`;
    modelId: string;
    signingAuthority: string;
  };
  application: PackRerankApplicationBinding;
  target: { targetId: string; targetPlanDigest: `sha256:${string}` };
  lifecycle: {
    releaseVersion: string;
    previousPackId: string | null;
    previousSemanticRoot: `sha256:${string}` | null;
  };
  revocation: DopplerPackV2['release']['revocation'];
  evidence: DopplerRerankEvidence;
  receiptDigest: `sha256:${string}`;
}

export declare function executePackRerank(args: {
  pack: DopplerPackV2;
  targetPlan: TargetPlan;
  targetPlanDigest: `sha256:${string}`;
  program: { rerank(request: Omit<PackRerankRequest, 'application'>): Promise<DopplerRerankEvidence> };
  request: PackRerankRequest;
}): Promise<PackRerankReceipt>;
