import type { ElectronReleaseStateCoordinator } from './release-state.js';
import type { DopplerRuntimeSession, CapsuleSessionOptions } from '../runtime/composition-root.js';
import type { CapsuleRerankRequest, CapsuleRerankReceipt } from '../runtime/capsule-rerank.js';

export type ElectronCapsuleOpenOptions = CapsuleSessionOptions & Record<string, unknown> & { signal?: AbortSignal };

export interface ElectronRendererRuntime {
  /** Caller owns this session and must close it; authorization is checked at open. */
  openCurrent(options?: ElectronCapsuleOpenOptions): Promise<DopplerRuntimeSession>;
  rerank(
    request: CapsuleRerankRequest,
    options?: ElectronCapsuleOpenOptions
  ): Promise<CapsuleRerankReceipt>;
}

export declare function createElectronRendererRuntime(options: {
  releaseState: Pick<ElectronReleaseStateCoordinator, 'resolveCurrent'>;
  openCapsule(capsulePath: string, options?: ElectronCapsuleOpenOptions): Promise<DopplerRuntimeSession>;
}): ElectronRendererRuntime;
