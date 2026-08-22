import type { TargetPlan } from '../../config/target-plan.js';
import type { DeviceProfile } from './target-selector.js';
import type { ResourceBinder } from './resource-binder.js';
import type { CommandExecutor } from './command-executor.js';
import type { SessionController, GenerationRunOptions } from './session-controller.js';

export const RUNTIME_CORE_VERSION = '1.0.0';

export interface RuntimePorts {
  device: {
    getProfile?: () => Promise<DeviceProfile> | DeviceProfile;
    hasF16?: boolean;
    hasSubgroups?: boolean;
  };
  packSource?: {
    fetchPack?: (id: string, options?: Record<string, unknown>) => Promise<unknown>;
  } | null;
  artifactStore?: unknown;
  cache?: unknown;
  observer?: unknown;
}

export interface DopplerRuntimeSession {
  modelId: string;
  bundleId?: string;
  selectedTargetId: string;
  selectedPlan: TargetPlan;
  deviceProfile: DeviceProfile;
  generate(options?: GenerationRunOptions): AsyncGenerator<number, void, void>;
}

export interface DopplerRuntime {
  version: string;
  ports: RuntimePorts;
  units: {
    resourceBinder: ResourceBinder;
    commandExecutor: CommandExecutor;
    sessionController: SessionController;
  };
  openPack(packOrId: string | unknown, options?: Record<string, unknown>): Promise<DopplerRuntimeSession>;
}

export declare function createDopplerRuntime(ports: RuntimePorts): DopplerRuntime;
