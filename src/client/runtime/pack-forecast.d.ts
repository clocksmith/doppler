import type { PackReleaseContract } from '../../config/pack-release-contract.js';
export interface PackForecastRequest {
  application: PackReleaseContract['application'];
  context: number[];
  horizon: number;
  assignmentHash: string | null;
  signal?: AbortSignal;
}
export interface PackForecastResult {
  horizon: number;
  quantileLevels: number[];
  layout: 'time-quantile';
  values: number[];
  receipt: Record<string, unknown> & { receiptDigest: string; inputHash: string; outputHash: string };
}
export declare function executePackForecast(options: Record<string, unknown>): Promise<PackForecastResult>;
