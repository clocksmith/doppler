export interface MoEVendorProfile {
  preferVec4Dequant: boolean;
  dequantTileShape: 'vec4' | 'scalar' | null;
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
  modelType: string;
  moeProfile?: MoEExecutionProfile;
}

export interface MoEExecutionProfile {
  id: string;
  label: string;
  expertExecutor: 'gpt-oss' | 'gemma4-packed' | 'mixtral' | string;
  intermediateSizeSource: 'architecture' | 'expert' | string;
  requiresShaderF16: boolean;
  routerScaleMode: 'none' | 'optional' | 'required';
  topkRouteExecutor: 'gemma4-route' | string | null;
  vendorProfile: Record<string, unknown>;
  kernelPathProfileResolver: 'gpt-oss' | 'mixtral' | string | null;
  kernelRuleModelType: string | null;
  shapePolicy: Record<string, unknown> | null;
}

export declare function resolveMoeExecutionProfile(
  config: Record<string, unknown>,
  options?: Record<string, unknown>
): MoEExecutionProfile;

export declare function resolveMoeIntermediateSize(
  config: Record<string, unknown>,
  moeProfile: MoEExecutionProfile
): number;

export declare function resolveMoeVendorProfile(moeProfile: MoEExecutionProfile): MoEVendorProfile;

export declare function validateMoeShape(config: MoEShapeConfig, options?: ValidateMoeShapeOptions): void;

export interface GptOssKernelPathProfile {
  routerTopK: string;
  dequantExpert: string;
}

export declare function resolveGptOssKernelPathProfile(
  context: Record<string, unknown>
): Promise<GptOssKernelPathProfile>;

export interface MixtralKernelPathProfile {
  routerTopK: string;
  dequantExpert: string;
}

export declare function resolveMixtralKernelPathProfile(
  context: Record<string, unknown>
): Promise<MixtralKernelPathProfile>;

export declare function resolveMoeKernelPathProfile(
  moeProfile: MoEExecutionProfile,
  context: Record<string, unknown>
): Promise<GptOssKernelPathProfile | MixtralKernelPathProfile | null>;
