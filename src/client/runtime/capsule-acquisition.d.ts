import type { CapsuleLoadingPolicy } from '../../config/capsule-loading.js';
import type { CapsuleV2Artifact } from '../../config/capsule-v2.js';
export interface CapsuleLoadProgress {
  phase: 'metadata' | 'artifact'; artifactId: string | null; loadedBytes: number; totalBytes: number | null;
}
export interface CapsuleAcquisitionOptions extends Partial<CapsuleLoadingPolicy> {
  signal?: AbortSignal | null;
  onLoadProgress?: ((event: CapsuleLoadProgress) => void) | null;
}
export interface CapsuleArtifactStreamOptions extends CapsuleAcquisitionOptions {
  /** Maximum backing-buffer size of each yielded chunk. No read-ahead is requested. */
  maxChunkBytes: number;
}
export interface CapsuleArtifactReader {
  readArtifact(artifact: CapsuleV2Artifact, options?: CapsuleAcquisitionOptions): Promise<Uint8Array | ArrayBuffer>;
  /** Ordered bytes; honor signal, release on return, and do not mutate until the next pull. */
  streamArtifact?(artifact: CapsuleV2Artifact, options: CapsuleArtifactStreamOptions): AsyncIterable<Uint8Array>;
}
export declare function assertCapsuleLoadActive(signal?: AbortSignal | null): void;
export declare function createCapsuleLoadScope(options?: CapsuleAcquisitionOptions): {
  options: CapsuleAcquisitionOptions & CapsuleLoadingPolicy & { signal: AbortSignal };
  abort(reason: unknown): void; close(): void;
};
export declare function waitForCapsuleRead<T>(task: Promise<T>, signal?: AbortSignal | null): Promise<T>;
export declare function fetchCapsuleBytes(url: string, options: CapsuleAcquisitionOptions, descriptor: {
  phase: CapsuleLoadProgress['phase']; artifactId: string | null; sizeBytes: number | null; maxBytes: number;
}): Promise<Uint8Array>;
export declare function fetchCapsuleMetadata(url: string, options: CapsuleAcquisitionOptions): Promise<unknown>;
