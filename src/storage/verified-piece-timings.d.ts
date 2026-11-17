import type { VerifiedPieceJobTimings, VerifiedRangeLeaseTimings } from './verified-piece-gather.js';
/** Undefined denotes a legacy/unsupported port. Present metadata must contain exactly finite nonnegative data fields. */
export declare function snapshotVerifiedPieceJobTimings(value: unknown): Readonly<VerifiedPieceJobTimings> | undefined;
export declare function snapshotVerifiedRangeLeaseTimings(value: unknown): Readonly<VerifiedRangeLeaseTimings> | undefined;
