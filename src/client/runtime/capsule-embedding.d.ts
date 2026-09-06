import type { CapsuleReleaseContract } from '../../config/capsule-release-contract.js';
import type { DopplerEmbeddingEvidence } from './model-session.js';
import type { CapsuleReleaseAuthorization } from '../../config/capsule-release-events.js';

export interface CapsuleEmbeddingRequest {
  application: CapsuleReleaseContract['application'];
  text: string;
  options?: { signal?: AbortSignal };
}

export interface CapsuleEmbeddingResult {
  readonly embedding: readonly number[];
  readonly tokens: readonly number[];
  readonly seqLen: number;
  readonly embeddingMode: 'mean' | 'last';
  readonly receipt: Readonly<Record<string, unknown> & {
    schema: 'doppler.capsule-execution-receipt/v1';
    operation: 'embed';
    receiptDigest: string;
    releaseAuthorization?: CapsuleReleaseAuthorization;
    inputHash: string;
    outputHash: string;
    resolution: DopplerEmbeddingEvidence['resolution'];
    executionIdentity: DopplerEmbeddingEvidence['executionIdentity'];
    backendIdentityHash: string;
  }>;
}

export declare function executeCapsuleEmbedding(options: Record<string, unknown>): Promise<CapsuleEmbeddingResult>;
