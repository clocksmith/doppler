/** Doppler-owned host adapter, compatible with the existing provider-v1 wire contract. */
export declare const NODE_WEBGPU_PROVIDER_SCHEMA: 'doe.webgpu-provider/v1';
export type NodeWebGPUGlobalMode = 'none' | 'install-missing' | 'replace';

export interface NodeWebGPUGlobalProvider {
  id: string;
  kind: 'global';
}

export interface NodeWebGPUModuleProvider {
  id: string;
  kind: 'module';
  module: string;
  gpu: {
    kind: 'export' | 'factory';
    path: string;
    args?: unknown[];
    resultPath?: string | null;
  };
  globals: {
    GPUBufferUsage: string;
    GPUShaderStage: string;
    GPUMapMode: string;
    GPUTextureUsage: string;
  };
}

export type NodeWebGPUProvider = NodeWebGPUGlobalProvider | NodeWebGPUModuleProvider;

export interface NodeWebGPUProviderOptions {
  providers: NodeWebGPUProvider[];
  adapterOptions: GPURequestAdapterOptions | null;
  globals: { mode: NodeWebGPUGlobalMode };
}

export interface NodeWebGPUProviderReceipt {
  schema: 'doe.webgpu-provider-receipt/v1';
  contract: 'doe.webgpu-provider/v1';
  providers: NodeWebGPUProvider[];
  providerOrder: string[];
  adapterOptions: unknown;
  globals: {
    mode: NodeWebGPUGlobalMode;
    installed: string[];
    restored: boolean;
  };
  attempts: Array<Record<string, unknown>>;
  selectedProviderId: string | null;
  ok: boolean;
  implementation?: 'doppler';
}

export interface NodeWebGPUProviderSession {
  readonly gpu: GPU | null;
  readonly adapter: GPUAdapter | null;
  module: unknown;
  receipt: NodeWebGPUProviderReceipt;
  close(): Promise<void>;
}

export declare function openNodeWebGPU(options: NodeWebGPUProviderOptions): Promise<NodeWebGPUProviderSession>;
