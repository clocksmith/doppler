export interface ConversionReportResultSchema {
  modelType: string | null;
  outputDir: string | null;
  shardCount: number | null;
  tensorCount: number | null;
  totalSize: number | null;
}

export interface ConversionReportManifestSchema {
  quantization: string | null;
  quantizationInfo: Record<string, unknown> | null;
  inference: {
    schema: string | null;
  } | null;
}

export interface ConversionReportSchema {
  schemaVersion: 1;
  suite: 'convert';
  command: 'convert';
  modelId: string;
  timestamp: string;
  source: 'doppler';
  result: ConversionReportResultSchema;
  manifest: ConversionReportManifestSchema | null;
  executionContractArtifact: Record<string, unknown> | null;
  layerPatternContractArtifact: Record<string, unknown> | null;
  requiredInferenceFieldsArtifact: Record<string, unknown> | null;
}

export declare const CONVERSION_REPORT_SCHEMA_VERSION: 1;
export declare const DEFAULT_CONVERSION_REPORT: ConversionReportSchema;

export declare function validateConversionReport(
  report: Record<string, unknown>
): ConversionReportSchema;
