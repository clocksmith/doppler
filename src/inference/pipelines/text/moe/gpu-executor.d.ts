export interface PerExpertScaleBufferResolution {
  buffer: GPUBuffer | null;
  ownedBuffer: GPUBuffer | null;
}

export declare function resolvePerExpertScaleBuffer(
  device: GPUDevice,
  value: unknown
): PerExpertScaleBufferResolution;
export declare function runGemma4RouteExperts(
  args: Record<string, unknown>
): Promise<{ buffer: GPUBuffer; dtype?: string }>;
export declare function runGptOssExpert(...args: unknown[]): Promise<void>;
export declare function runGemma4Expert(...args: unknown[]): Promise<void>;
export declare function runMixtralExpert(...args: unknown[]): Promise<void>;

