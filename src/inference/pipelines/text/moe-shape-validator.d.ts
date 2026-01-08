export interface MoEVendorProfile {
  preferVec4Dequant: boolean;
  dequantTileShape: 'vec4' | 'scalar';
  routerWorkgroupSize: number;
  maxTokensPerExpertScale: number;
}

export interface MoEShapeConfig {
  hiddenSize: number;
  intermediateSize: number;
  moeTopK: number;
  numExperts: number;
  expertFormat?: string | null;
}

export interface ValidateMoeShapeOptions {
  modelType?: string;
}

export declare function resolveMoeVendorProfile(modelType: string): MoEVendorProfile;

export declare function validateMoeShape(config: MoEShapeConfig, options?: ValidateMoeShapeOptions): void;

export interface GptOssKernelPathProfile {
  routerTopK: string;
  dequantExpert: string;
}

export declare function resolveGptOssKernelPathProfile(
  context: Record<string, unknown>
): Promise<GptOssKernelPathProfile>;
