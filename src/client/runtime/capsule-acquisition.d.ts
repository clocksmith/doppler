import type { CapsuleLoadingPolicy } from '../../config/capsule-loading.js';
export interface CapsuleLoadProgress {
  phase: 'metadata' | 'artifact'; artifactId: string | null; loadedBytes: number; totalBytes: number | null;
}
export interface CapsuleAcquisitionOptions extends Partial<CapsuleLoadingPolicy> {
  signal?: AbortSignal | null;
  onLoadProgress?: ((event: CapsuleLoadProgress) => void) | null;
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
