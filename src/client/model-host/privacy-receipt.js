/** Deliberately separate from local generation/inspection evidence, which contains private text and token IDs. */
import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';

/** Whitelist only requester validation metadata. Never copy arbitrary stats, hashes, errors or evidence.
 * @type {import('./privacy-receipt.js').buildProviderPrivacyReceipt} */
export function buildProviderPrivacyReceipt(input) {
  const { attemptId, policyId, routeId, modelIdentity, backendId, status } = input;
  for (const identity of [policyId, routeId, modelIdentity]) {
    if (typeof identity !== 'string' || !/^sha256:[a-f0-9]{64}$/.test(identity)) {
      throw new Error('Privacy receipt requires policy, route and model digests.');
    }
  }
  for (const id of [attemptId, backendId]) {
    if (typeof id !== 'string' || !id || id.length > 256) throw new Error('Privacy receipt requires bounded identifiers.');
  }
  if (!['completed', 'cancelled', 'failed', 'unknown'].includes(status)) throw new Error('Privacy receipt requires explicit outcome.');
  /** @type {Omit<import('./privacy-receipt.js').ProviderPrivacyReceipt, 'receiptId'>} */
  const receipt = { schema: 'doppler.provider-privacy-receipt/v1', attemptId, policyId, routeId, modelIdentity, backendId, status,
    evidence: 'execution-record', confidentialityProof: false, integrityProof: false, availabilityProof: false,
    publicationAuthorized: false, reuseAuthorized: false };
  return Object.freeze({ ...receipt, receiptId: computeCanonicalSha256(receipt) });
}
