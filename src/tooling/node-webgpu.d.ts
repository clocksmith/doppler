import type {
  NodeWebGPUGlobalMode,
  NodeWebGPUModuleProvider,
  NodeWebGPUProvider,
  NodeWebGPUProviderOptions,
  NodeWebGPUProviderReceipt,
  NodeWebGPUProviderSession,
} from './provider-v1-contract.js';

export class DopplerNodeWebGPUError extends Error {
  readonly code: string;
  readonly stage: string;
  readonly receipt: NodeWebGPUProviderReceipt | null;
}

export interface DopplerNodeWebGPUContractOptions {
  providerContractModule?: string;
}

export interface BootstrapNodeWebGPUOptions extends DopplerNodeWebGPUContractOptions {
  providerOptions?: NodeWebGPUProviderOptions;
}

export interface BootstrapNodeWebGPUProviderOptions extends DopplerNodeWebGPUContractOptions {
  provider?: NodeWebGPUProvider;
  id?: string;
  gpu?: NodeWebGPUModuleProvider['gpu'];
  globals?: NodeWebGPUModuleProvider['globals'];
  createArgs?: unknown[];
  adapterOptions?: GPURequestAdapterOptions | null;
  globalMode?: NodeWebGPUGlobalMode;
}

export interface BootstrapNodeWebGPUResult {
  ok: boolean;
  provider: string | null;
  detail: string | null;
  module: unknown;
  session: NodeWebGPUProviderSession | null;
  receipt: NodeWebGPUProviderReceipt | null;
  error?: Error;
}

export interface ReleaseNodeWebGPUResult {
  released: boolean;
  provider: string | null;
  reason: 'not-owned' | null;
  receipt: NodeWebGPUProviderReceipt | null;
}

export declare function openNodeWebGPU(
  providerOptions: NodeWebGPUProviderOptions,
  options?: DopplerNodeWebGPUContractOptions,
): Promise<NodeWebGPUProviderSession>;

export declare function bootstrapNodeWebGPU(
  options?: BootstrapNodeWebGPUOptions,
): Promise<BootstrapNodeWebGPUResult>;

export declare function releaseNodeWebGPU(): Promise<ReleaseNodeWebGPUResult>;

export declare function bootstrapNodeWebGPUProvider(
  providerSpecifier: string,
  options?: BootstrapNodeWebGPUProviderOptions,
): Promise<BootstrapNodeWebGPUResult>;
