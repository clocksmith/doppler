export interface BootstrapNodeWebGPUResult {
  ok: boolean;
  provider: string | null;
  module?: Record<string, unknown> | null;
}

export interface BootstrapNodeWebGPUProviderOptions {
  force?: boolean;
}

export declare function bootstrapNodeWebGPU(): Promise<BootstrapNodeWebGPUResult>;

export declare function bootstrapNodeWebGPUProvider(
  providerSpecifier: string,
  options?: BootstrapNodeWebGPUProviderOptions
): Promise<BootstrapNodeWebGPUResult>;
