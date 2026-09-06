import type { CapsuleReleaseAuthorization } from '../../config/capsule-release-events.js';
export declare function createCapsuleReleaseAuthorization(lifecycle: { authorization: CapsuleReleaseAuthorization } | null): {
  receiptFields: { releaseAuthorization?: CapsuleReleaseAuthorization };
  assertAssignment(assignment: unknown): void;
  bindReceipt<T extends { receiptDigest: string }>(receipt: T): T & { releaseAuthorization?: CapsuleReleaseAuthorization };
  bindResult<T extends { receipt: { receiptDigest: string } }>(result: T): T;
};
