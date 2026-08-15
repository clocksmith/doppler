import type { DopplerRevocationRegistry } from './revocation-policy.js';

export interface DopplerRevocationPublicKey {
  id: string;
  publicKeyJwk: JsonWebKey;
}

export interface DopplerRevocationStateStore {
  load(): Promise<unknown | null>;
  save(state: unknown): Promise<void>;
}

export interface DopplerSignedRevocationAuthorityOptions {
  authorityId: string;
  url: string;
  initialEpoch: number;
  onlineKeys: DopplerRevocationPublicKey[];
  recoveryKeys: DopplerRevocationPublicKey[];
  refreshIntervalMs: number;
  requestTimeoutMs: number;
  maxBytes: number;
  maxClockSkewMs: number;
  maxEnvelopeLifetimeMs: number;
  stateStore: DopplerRevocationStateStore;
  fetchFn?: typeof fetch;
  now?: () => number;
}

export interface DopplerSignedRevocationEnvelope {
  schema: 'doppler.signed-revocation-envelope/v1';
  authorityId: string;
  epoch: number;
  sequence: number;
  issuedAtUtc: string;
  expiresAtUtc: string;
  registry: DopplerRevocationRegistry;
  keyring: {
    onlineKeys: DopplerRevocationPublicKey[];
    revokedKeys: DopplerRevocationPublicKey[];
  } | null;
  signerId: string;
  signature: string;
}

export interface DopplerSignedRevocationStatus {
  configured: boolean;
  authorityId?: string;
  epoch?: number;
  sequence?: number;
  expiresAtUtc?: string | null;
  signatureVerification: 'unavailable' | 'pending' | 'verified';
  current?: boolean;
  offline?: boolean;
  lastError?: string | null;
}

export declare function serializeSignedRevocationEnvelope(envelope: DopplerSignedRevocationEnvelope): string;
export declare function configureSignedRevocationAuthority(options: DopplerSignedRevocationAuthorityOptions): Promise<DopplerSignedRevocationStatus>;
export declare function refreshSignedRevocations(options?: { force?: boolean }): Promise<DopplerSignedRevocationStatus>;
export declare function getSignedRevocationStatus(): DopplerSignedRevocationStatus;
