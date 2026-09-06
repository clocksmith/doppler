import type { CapsuleReleaseContract } from '../../config/capsule-release-contract.js';
import type { CapsuleReleaseAuthorization } from '../../config/capsule-release-events.js';
export interface CapsuleForecastRequest {
  application: CapsuleReleaseContract['application'];
  context: number[];
  horizon: number;
  assignmentHash: string | null;
  signal?: AbortSignal;
}
export interface CapsuleForecastResult {
  horizon: number;
  quantileLevels: number[];
  layout: 'time-quantile';
  values: number[];
  receipt: Record<string, unknown> & { receiptDigest: string; inputHash: string; outputHash: string; releaseAuthorization?: CapsuleReleaseAuthorization };
}
export declare function executeCapsuleForecast(options: Record<string, unknown>): Promise<CapsuleForecastResult>;
