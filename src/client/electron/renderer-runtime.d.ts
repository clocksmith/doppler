import type { ElectronReleaseStateCoordinator } from './release-state.js';
import type { DopplerRuntimeSession } from '../runtime/composition-root.js';
import type { PackRerankRequest, PackRerankReceipt } from '../runtime/pack-rerank.js';

export type ElectronPackOpenOptions = Record<string, unknown> & { signal?: AbortSignal };

export interface ElectronRendererRuntime {
  /** Caller owns this session and must close it; authorization is checked at open. */
  openCurrent(options?: ElectronPackOpenOptions): Promise<DopplerRuntimeSession>;
  rerank(
    request: PackRerankRequest,
    options?: ElectronPackOpenOptions
  ): Promise<PackRerankReceipt>;
}

export declare function createElectronRendererRuntime(options: {
  releaseState: Pick<ElectronReleaseStateCoordinator, 'resolveCurrent'>;
  openPack(packPath: string, options?: ElectronPackOpenOptions): Promise<DopplerRuntimeSession>;
}): ElectronRendererRuntime;
