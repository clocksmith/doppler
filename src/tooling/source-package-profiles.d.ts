export interface DirectSourcePackageRuntimeProfile {
  modelType: string;
  architecture: Record<string, unknown>;
  rawConfig: Record<string, unknown>;
  manifestConfig: Record<string, unknown>;
  manifestInference: Record<string, unknown>;
  tokenizer?: {
    task?: Record<string, unknown> | null;
    litertlm?: Record<string, unknown> | null;
  } | null;
}

export interface DirectSourcePackageProfile {
  id: string;
  runtime: DirectSourcePackageRuntimeProfile;
  package: {
    task?: {
      tfliteEntry?: string | null;
      tokenizerModelEntry?: string | null;
      metadataEntry?: string | null;
    } | null;
    litertlm?: {
      tfliteModelType?: string | null;
      tokenizerSectionType?: string | null;
      metadataSectionType?: string | null;
    } | null;
  } | null;
}

export declare function resolveDirectSourcePackageProfile(options?: {
  sourceKind?: string | null;
  packageBasename?: string | null;
}): DirectSourcePackageProfile | null;
