export const MANIFEST_CONVERSION_PREFLIGHT_SCHEMA_ID: 'doppler.manifest-conversion-preflight/v1';

export declare function createManifestConversionPreflightReceipt(options: {
  rawConfig: Record<string, unknown>;
  conversionConfig: Record<string, unknown>;
  semanticReceipt: Record<string, unknown>;
  headers: Record<string, unknown>;
  weightIndex: Record<string, unknown>;
  tensorPolicy: Record<string, unknown>;
  tensorClosureReceipt: Record<string, unknown>;
  sourceAcquisitionReceipt: Record<string, unknown>;
  policy: Record<string, unknown>;
}): Record<string, unknown>;
