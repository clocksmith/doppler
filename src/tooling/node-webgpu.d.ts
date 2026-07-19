export interface BootstrapNodeWebGPUResult {
  ok: boolean;
  provider: string | null;
  detail?: string | null;
  module?: Record<string, unknown> | null;
}

export interface ReleaseNodeWebGPUResult {
  released: boolean;
  provider: string | null;
  reason: 'not-owned' | 'provider-replaced' | null;
}

export interface BootstrapNodeWebGPUProviderOptions {
  force?: boolean;
}

export declare function bootstrapNodeWebGPU(): Promise<BootstrapNodeWebGPUResult>;

export declare function releaseNodeWebGPU(): ReleaseNodeWebGPUResult;

export declare function bootstrapNodeWebGPUProvider(
  providerSpecifier: string,
  options?: BootstrapNodeWebGPUProviderOptions
): Promise<BootstrapNodeWebGPUResult>;
