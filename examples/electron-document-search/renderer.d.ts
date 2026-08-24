import type { ElectronReleaseStateCoordinator, ElectronRendererRuntime } from '../../src/client/electron/index.js';

export declare function createDocumentSearchRenderer(
  releaseState: Pick<ElectronReleaseStateCoordinator, 'resolveCurrent'>
): ElectronRendererRuntime;
