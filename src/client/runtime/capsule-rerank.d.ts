import type { DopplerCapsuleV2 } from '../../config/capsule-v2.js';
import type { TargetPlan } from '../../config/target-plan.js';
import type { DopplerRerankEvidence } from './model-session.js';
import type { CapsuleReleaseAuthorization } from '../../config/capsule-release-events.js';

export const CAPSULE_RERANK_RECEIPT_SCHEMA: 'doppler.capsule-rerank-receipt/v1';

export interface CapsuleRerankApplicationBinding {
  applicationId: string;
  applicationRevision: string;
  applicationRevisionDigest: `sha256:${string}`;
  workload: { id: string; digest: `sha256:${string}` };
  oracle: { id: string; digest: `sha256:${string}` };
}

export interface CapsuleRerankRequest {
  application: CapsuleRerankApplicationBinding;
  query: string;
  documents: string[];
  options?: { benchmark?: boolean; signal?: AbortSignal };
}

export interface CapsuleRerankReceipt {
  schema: 'doppler.capsule-rerank-receipt/v1';
  capsule: {
    capsuleId: string;
    semanticRoot: `sha256:${string}`;
    modelId: string;
    signingAuthority: string;
  };
  application: CapsuleRerankApplicationBinding;
  target: { targetId: string; targetPlanDigest: `sha256:${string}` };
  lifecycle: {
    releaseVersion: string;
    previousCapsuleId: string | null;
    previousSemanticRoot: `sha256:${string}` | null;
  };
  revocation: DopplerCapsuleV2['release']['revocation'];
  releaseAuthorization?: CapsuleReleaseAuthorization;
  evidence: DopplerRerankEvidence;
  receiptDigest: `sha256:${string}`;
}

export declare function executeCapsuleRerank(args: {
  capsule: DopplerCapsuleV2;
  targetPlan: TargetPlan;
  targetPlanDigest: `sha256:${string}`;
  program: { rerank(request: Omit<CapsuleRerankRequest, 'application'>): Promise<DopplerRerankEvidence> };
  request: CapsuleRerankRequest;
}): Promise<CapsuleRerankReceipt>;
