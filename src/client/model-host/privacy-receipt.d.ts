export interface ProviderPrivacyReceiptInput {
  attemptId: string; policyId: string; routeId: string; modelIdentity: string; backendId: string;
  status: 'completed' | 'cancelled' | 'failed' | 'unknown';
}
export interface ProviderPrivacyReceipt extends ProviderPrivacyReceiptInput {
  schema: 'doppler.provider-privacy-receipt/v1'; receiptId: string; evidence: 'execution-record';
  confidentialityProof: false; integrityProof: false; availabilityProof: false; publicationAuthorized: false; reuseAuthorized: false;
}
export function buildProviderPrivacyReceipt(input: ProviderPrivacyReceiptInput): Readonly<ProviderPrivacyReceipt>;
