export declare const APPLICATION_GATE_RECEIPT_SCHEMA: 'doppler.application-gate-receipt/v1';
export declare const ELECTRON_FLEET_RECEIPT_SCHEMA: 'doppler.electron-fleet-receipt/v1';
export declare const RELEASE_DECISION_SCHEMA: 'doppler.release-decision/v1';
export declare const RELEASE_FAILURE_BUNDLE_SCHEMA: 'doppler.release-failure-bundle/v1';

export interface ReleaseEvidenceSigner {
  authority: string;
  privateKeyJwk: JsonWebKey;
  publicKeyJwk: JsonWebKey;
}

export declare function hashProductionReleaseEvidence(value: Record<string, unknown>): `sha256:${string}`;
export declare function validateApplicationGateReceipt(value: unknown): { ok: boolean; errors: string[] };
export declare function validateElectronFleetReceipt(value: unknown): { ok: boolean; errors: string[] };
export declare function validateReleaseDecision(value: unknown): { ok: boolean; errors: string[] };
export declare function signProductionReleaseEvidence(
  value: Record<string, unknown>,
  signer: ReleaseEvidenceSigner
): Promise<Record<string, unknown>>;
export declare function verifyProductionReleaseEvidenceSignature(
  value: Record<string, unknown>,
  trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>
): Promise<true>;
