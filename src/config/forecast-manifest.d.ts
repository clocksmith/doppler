import type { DopplerPack } from './pack.js';
import type { TargetPlan } from './target-plan.js';
export const FORECAST_MANIFEST_SCHEMA: 'doppler.forecast-manifest/v1';
export interface ForecastContract {
  contextLength: number;
  predictionLength: number;
  quantiles: number[];
  inputDtype: 'f32';
  outputDtype: 'f32';
  outputLayout: 'time-quantile';
  missingInput: 'left-pad-masked-zero';
  inputSlot: string;
  maskSlot: string;
  requestSlot: string;
  outputSlot: string;
}
export interface ForecastManifest {
  schema: 'doppler.forecast-manifest/v1';
  modelId: string;
  forecast: ForecastContract;
  executionGraphHash: string;
  execution: { steps: Array<Record<string, unknown>> };
  uploads: Array<{ slotId: string; artifactId: string; offsetBytes: number; sizeBytes: number }>;
}
export declare function validateForecastManifest(manifest: unknown, pack: DopplerPack, targetPlan: TargetPlan): ForecastContract;
