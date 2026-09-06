import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';
import { freezeCapsuleV2 } from '../../config/capsule-v2.js';

// One receipt/assignment boundary for every Capsule operation; no model policy lives here.
export function createCapsuleReleaseAuthorization(lifecycle) {
  const authorization = lifecycle?.authorization;
  function bindReceipt(receipt) {
    if (!authorization) return receipt;
    const { receiptDigest, ...payload } = receipt;
    const bound = { ...payload, releaseAuthorization: authorization };
    return freezeCapsuleV2({ ...bound, receiptDigest: computeCanonicalSha256(bound) });
  }
  return {
    receiptFields: authorization ? { releaseAuthorization: authorization } : {},
    assertAssignment(assignment) {
      if (authorization?.mode === 'retained-local' && assignment != null) {
        throw new Error('Retained local use does not authorize delegated assignments.');
      }
    },
    bindReceipt,
    bindResult(result) {
      return authorization ? Object.freeze({ ...result, receipt: bindReceipt(result.receipt) }) : result;
    },
  };
}
