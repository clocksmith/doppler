import type { ElectronReleaseStateStore } from 'doppler-gpu/electron';
import type { DopplerPackV3, PackReleaseEvent, PackReleasePolicy, ReleaseCheckpoint } from 'doppler-gpu/pack';

export interface DocumentSearchCheckpointStore {
  load(): Promise<ReleaseCheckpoint | null>;
  compareAndSwap(expectedSequence: number, next: ReleaseCheckpoint): Promise<boolean>;
}

export declare function createDocumentSearchReleaseStore(filename: string): ElectronReleaseStateStore;
export declare function createDocumentSearchCheckpointStore(filename: string): DocumentSearchCheckpointStore;
export declare function prepareDocumentSearchReleaseOptions(options: {
  pack: DopplerPackV3;
  releaseEvents: PackReleaseEvent[];
  releaseTrustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  checkpointStore: DocumentSearchCheckpointStore;
  minimumSequence: number;
  now: string;
}): Promise<{
  releaseEvents: PackReleaseEvent[];
  releaseTrustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  releasePolicy: PackReleasePolicy;
  persistReleaseCheckpoint(checkpoint: ReleaseCheckpoint): Promise<void>;
}>;
