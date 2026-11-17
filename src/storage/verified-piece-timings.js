/** @param {unknown} value @param {boolean} roundTrip */
function snapshot(value, roundTrip) {
  if (value === undefined) return undefined;
  const fields = ['workerWallMs', 'readWaitMs', 'verificationMs', 'copyMs', ...(roundTrip ? ['roundTripMs'] : [])];
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).length !== fields.length) {
    throw new Error('Invalid verified source timing metadata.');
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  if (fields.some(field => !descriptors[field] || !('value' in descriptors[field])
    || typeof descriptors[field].value !== 'number' || !Number.isFinite(descriptors[field].value)
    || descriptors[field].value < 0 || descriptors[field].value > Number.MAX_SAFE_INTEGER)) {
    throw new Error('Invalid verified source timing metadata.');
  }
  return Object.freeze({ workerWallMs: descriptors.workerWallMs.value, readWaitMs: descriptors.readWaitMs.value,
    verificationMs: descriptors.verificationMs.value, copyMs: descriptors.copyMs.value,
    ...(roundTrip ? { roundTripMs: descriptors.roundTripMs.value } : {}) });
}

/** @type {typeof import('./verified-piece-timings.js').snapshotVerifiedPieceJobTimings} */
export function snapshotVerifiedPieceJobTimings(value) { return snapshot(value, false); }

/** @type {typeof import('./verified-piece-timings.js').snapshotVerifiedRangeLeaseTimings} */
export function snapshotVerifiedRangeLeaseTimings(value) {
  return /** @type {Readonly<import('./verified-piece-gather.js').VerifiedRangeLeaseTimings> | undefined} */ (snapshot(value, true));
}
