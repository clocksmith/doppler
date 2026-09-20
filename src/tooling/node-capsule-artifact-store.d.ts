import type { CapsuleV2Artifact } from '../config/capsule-v2.js';
import type { CapsuleArtifactStreamOptions } from '../client/runtime/capsule-acquisition.js';

export interface NodeCapsuleArtifactStore {
  streamArtifact(artifact: CapsuleV2Artifact, options: CapsuleArtifactStreamOptions): AsyncGenerator<Uint8Array>;
  hashArtifact(artifact: CapsuleV2Artifact, options?: { signal?: AbortSignal | null }): Promise<{ hash: string; sizeBytes: number }>;
  readArtifact(artifact: CapsuleV2Artifact, options?: { signal?: AbortSignal | null; onLoadProgress?: ((event: {
    phase: 'artifact'; artifactId: string; loadedBytes: number; totalBytes: number;
  }) => void) | null }): Promise<Uint8Array>;
  resolveArtifactPath(artifact: CapsuleV2Artifact): string;
  resolveArtifactUrl(artifact: CapsuleV2Artifact): string;
}

export declare function createNodeCapsuleArtifactStore(capsulePath: string): NodeCapsuleArtifactStore;
