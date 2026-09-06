import type { ElectronReleaseStateCoordinator, ElectronRendererRuntime } from 'doppler-gpu/electron';
import type { RuntimePorts } from 'doppler-gpu';
import type { DopplerCapsuleOpenOptions } from 'doppler-gpu/host';

export declare function createDocumentSearchHostRenderer(
  releaseState: Pick<ElectronReleaseStateCoordinator, 'resolveCurrent'>,
  trustOptions: DopplerCapsuleOpenOptions
): ElectronRendererRuntime;

export declare function createDocumentSearchRenderer(
  releaseState: Pick<ElectronReleaseStateCoordinator, 'resolveCurrent'>,
  runtimePorts: RuntimePorts
): ElectronRendererRuntime;
