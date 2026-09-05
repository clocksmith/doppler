import type { ElectronReleaseStateCoordinator, ElectronRendererRuntime } from 'doppler-gpu/electron';
import type { RuntimePorts } from 'doppler-gpu';

export declare function createDocumentSearchRenderer(
  releaseState: Pick<ElectronReleaseStateCoordinator, 'resolveCurrent'>,
  runtimePorts: RuntimePorts
): ElectronRendererRuntime;
