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
  runtime?: DirectSourcePackageRuntimeProfile | null;
  package: {
    task?: {
      tfliteEntry?: string | null;
      tokenizerModelEntry?: string | null;
      metadataEntry?: string | null;
      unsupported?: {
        code?: string | null;
        message?: string | null;
        recommendation?: string | null;
      } | null;
    } | null;
    litertlm?: {
      tfliteModelType?: string | null;
      embedderTFLiteModelType?: string | null;
      graphAdapter?: string | null;
      fixedInt4Scale?: number | null;
      fixedInt4StorageEncoding?: 'signed' | 'offset_binary' | null;
      layerScalarLayers?: number[] | null;
      missingLayerScalarValue?: number | null;
      executionTemplateProfileId?: string | null;
      tokenizerSectionType?: string | null;
      metadataSectionType?: string | null;
      unsupported?: {
        code?: string | null;
        message?: string | null;
        recommendation?: string | null;
      } | null;
    } | null;
  } | null;
}

export declare function resolveDirectSourcePackageProfile(options?: {
  sourceKind?: string | null;
  packageBasename?: string | null;
}): DirectSourcePackageProfile | null;
