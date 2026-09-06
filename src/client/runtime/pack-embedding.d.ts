import type { PackReleaseContract } from '../../config/pack-release-contract.js';
import type { DopplerEmbeddingEvidence } from './model-session.js';
import type { PackReleaseAuthorization } from '../../config/pack-release-events.js';

export interface PackEmbeddingRequest {
  application: PackReleaseContract['application'];
  text: string;
  options?: { signal?: AbortSignal };
}

export interface PackEmbeddingResult {
  readonly embedding: readonly number[];
  readonly tokens: readonly number[];
  readonly seqLen: number;
  readonly embeddingMode: 'mean' | 'last';
  readonly receipt: Readonly<Record<string, unknown> & {
    schema: 'doppler.pack-execution-receipt/v1';
    operation: 'embed';
    receiptDigest: string;
    releaseAuthorization?: PackReleaseAuthorization;
    inputHash: string;
    outputHash: string;
    resolution: DopplerEmbeddingEvidence['resolution'];
    executionIdentity: DopplerEmbeddingEvidence['executionIdentity'];
    backendIdentityHash: string;
  }>;
}

export declare function executePackEmbedding(options: Record<string, unknown>): Promise<PackEmbeddingResult>;
