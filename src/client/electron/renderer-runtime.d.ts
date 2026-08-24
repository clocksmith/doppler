import type { ElectronReleaseStateCoordinator } from './release-state.js';

export interface ElectronRendererRuntime {
  openCurrent(options?: Record<string, unknown> & { signal?: AbortSignal }): Promise<Record<string, unknown>>;
  rerank(
    query: string,
    documents: string[],
    options?: Record<string, unknown> & { signal?: AbortSignal }
  ): Promise<unknown>;
}

export declare function createElectronRendererRuntime(options: {
  releaseState: Pick<ElectronReleaseStateCoordinator, 'resolveCurrent'>;
  openPack(packPath: string, options?: Record<string, unknown>): Promise<Record<string, unknown>>;
}): ElectronRendererRuntime;
