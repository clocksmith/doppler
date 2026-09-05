import type { ElectronReleaseStateCoordinator, ElectronRendererRuntime } from 'doppler-gpu/electron';
import type { RuntimePorts } from 'doppler-gpu';
import type { DopplerPackOpenOptions } from 'doppler-gpu/host';

export declare function createDocumentSearchHostRenderer(
  releaseState: Pick<ElectronReleaseStateCoordinator, 'resolveCurrent'>,
  trustOptions: DopplerPackOpenOptions
): ElectronRendererRuntime;

export declare function createDocumentSearchRenderer(
  releaseState: Pick<ElectronReleaseStateCoordinator, 'resolveCurrent'>,
  runtimePorts: RuntimePorts
): ElectronRendererRuntime;
