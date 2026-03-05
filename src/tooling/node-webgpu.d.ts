export interface BootstrapNodeWebGPUResult {
  ok: boolean;
  provider: string | null;
}

export declare function bootstrapNodeWebGPU(): Promise<BootstrapNodeWebGPUResult>;
