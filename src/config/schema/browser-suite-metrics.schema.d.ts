export interface BrowserSuiteMetricsSchema {
  schemaVersion: 1;
  source: 'doppler';
  suite: string;
  executionContractArtifact: Record<string, unknown> | null;
  layerPatternContractArtifact: Record<string, unknown> | null;
  requiredInferenceFieldsArtifact: Record<string, unknown> | null;
  referenceTranscript: Record<string, unknown> | null;
  [key: string]: unknown;
}

export declare const BROWSER_SUITE_METRICS_SCHEMA_VERSION: 1;
export declare const DEFAULT_BROWSER_SUITE_METRICS: Readonly<BrowserSuiteMetricsSchema>;

export declare function validateBrowserSuiteMetrics(
  metrics: Record<string, unknown>
): BrowserSuiteMetricsSchema;
