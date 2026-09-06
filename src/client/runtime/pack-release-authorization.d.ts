import type { PackReleaseAuthorization } from '../../config/pack-release-events.js';
export declare function createPackReleaseAuthorization(lifecycle: { authorization: PackReleaseAuthorization } | null): {
  receiptFields: { releaseAuthorization?: PackReleaseAuthorization };
  assertAssignment(assignment: unknown): void;
  bindReceipt<T extends { receiptDigest: string }>(receipt: T): T & { releaseAuthorization?: PackReleaseAuthorization };
  bindResult<T extends { receipt: { receiptDigest: string } }>(result: T): T;
};
