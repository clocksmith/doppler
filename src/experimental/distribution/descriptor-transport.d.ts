export declare const DESCRIPTOR_TRANSPORT_CONTRACT_VERSION: 1;

export interface PeerCapabilityProfile {
  available_vram_bytes: number;
  backends: string[];
  supported_generators: string[];
  bandwidth_bps: number;
  latency_ms: number;
  reliability_score: number;
}

export interface DescriptorRequiredShard {
  role: 'kronecker_sum' | 'coordinate_inr' | 'sparse_outliers';
  file: string;
  hash: string | null;
}

export interface DescriptorMissingShard extends DescriptorRequiredShard {
  reason: 'descriptor_hash_mismatch' | 'not_cached' | 'shard_hash_mismatch';
}

export interface DescriptorShardNegotiationResult {
  contractVersion: number;
  descriptorHash: string | null;
  peerDescriptorHash: string | null;
  ready: boolean;
  requiredShards: DescriptorRequiredShard[];
  missingShards: DescriptorMissingShard[];
}

export interface ActivationTransportValidationResult {
  contractVersion: number;
  modelDim: number;
  tokenCount: number;
  bytesPerToken: number;
  expectedBytes: number;
  actualBytes: number;
}

export type DescriptorPeerAssignmentBlocker =
  | {
      code: 'descriptor_shards_missing';
      missingShards: DescriptorMissingShard[];
    }
  | {
      code: 'insufficient_vram';
      requiredVramBytes: number;
      availableVramBytes: number;
    };

export interface DescriptorPeerAssignmentResult {
  contractVersion: number;
  assignable: boolean;
  blockers: DescriptorPeerAssignmentBlocker[];
  profile: PeerCapabilityProfile;
  requiredGenerators: string[];
  cache: DescriptorShardNegotiationResult;
  requiredDownloads: DescriptorMissingShard[];
  activation: ActivationTransportValidationResult | null;
  requiredVramBytes: number | null;
}

export declare function normalizePeerCapabilityProfile(
  value: unknown,
  label?: string
): PeerCapabilityProfile;

export declare function getDescriptorRequiredGenerators(
  descriptorManifest: Record<string, unknown>
): string[];

export declare function assertPeerSupportsDescriptor(
  peerCapabilityProfile: unknown,
  descriptorManifest: Record<string, unknown>
): {
  profile: PeerCapabilityProfile;
  requiredGenerators: string[];
};

export declare function getDescriptorRequiredShards(
  descriptorManifest: Record<string, unknown>,
  descriptorShardHashes?: Record<string, string>
): DescriptorRequiredShard[];

export declare function negotiateDescriptorShardCache(options?: {
  descriptorManifest?: Record<string, unknown>;
  descriptorShardHashes?: Record<string, string>;
  peerDescriptorCache?: {
    descriptorHash?: string | null;
    descriptor_hash?: string | null;
    shards?: Record<string, string | { hash?: string | null }>;
  } | null;
}): DescriptorShardNegotiationResult;

export declare function validateActivationTransportPayload(
  payload: ArrayBuffer | Uint8Array,
  options: {
    modelDim: number;
    tokenCount: number;
  }
): ActivationTransportValidationResult;

export declare function createDescriptorPeerAssignment(options?: {
  descriptorManifest?: Record<string, unknown>;
  peerCapabilityProfile?: unknown;
  descriptorShardHashes?: Record<string, string>;
  peerDescriptorCache?: {
    descriptorHash?: string | null;
    descriptor_hash?: string | null;
    shards?: Record<string, string | { hash?: string | null }>;
  } | null;
  activationPayload?: ArrayBuffer | Uint8Array | null;
  modelDim?: number;
  tokenCount?: number;
  requiredVramBytes?: number | null;
  failClosed?: boolean;
}): DescriptorPeerAssignmentResult;
