import type { ElectronReleaseStateStore } from 'doppler-gpu/electron';
import type { DopplerCapsuleV3, CapsuleReleaseEvent, CapsuleReleasePolicy, ReleaseCheckpoint } from 'doppler-gpu/capsule';

export interface DocumentSearchCheckpointStore {
  load(): Promise<ReleaseCheckpoint | null>;
  compareAndSwap(expectedSequence: number, next: ReleaseCheckpoint): Promise<boolean>;
}

export declare function createDocumentSearchReleaseStore(filename: string): ElectronReleaseStateStore;
export declare function createDocumentSearchCheckpointStore(filename: string): DocumentSearchCheckpointStore;
export declare function prepareDocumentSearchReleaseOptions(options: {
  capsule: DopplerCapsuleV3;
  releaseEvents: CapsuleReleaseEvent[];
  releaseTrustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  checkpointStore: DocumentSearchCheckpointStore;
  minimumSequence: number;
  now: string;
  retainedLocalUse?: CapsuleReleasePolicy['retainedLocalUse'];
}): Promise<{
  releaseEvents: CapsuleReleaseEvent[];
  releaseTrustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  releasePolicy: CapsuleReleasePolicy;
  persistReleaseCheckpoint(checkpoint: ReleaseCheckpoint): Promise<void>;
}>;
