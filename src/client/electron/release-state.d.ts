export declare const ELECTRON_RELEASE_STATE_SCHEMA: 'doppler.electron-release-state/v1';
export declare const ELECTRON_REVOCATION_SNAPSHOT_SCHEMA: 'doppler.electron-revocation-snapshot/v1';

export interface ElectronPackReference {
  packId: string;
  semanticRoot: `sha256:${string}`;
  path: string;
}

export interface ElectronRevocationSnapshot extends Record<string, unknown> {
  schema: 'doppler.electron-revocation-snapshot/v1';
  authorityId: string;
  policyDigest: `sha256:${string}`;
  sequence: number;
  issuedAtUtc: string;
  expiresAtUtc: string;
  revokedSemanticRoots: Array<`sha256:${string}`>;
  digest: `sha256:${string}`;
  signature: Record<string, unknown>;
}

export interface ElectronReleaseStateStore {
  load(): Promise<unknown | null>;
  compareAndSwap(expectedSequence: number, nextState: ElectronReleaseState): Promise<boolean>;
}

export interface ElectronReleaseState {
  schema: 'doppler.electron-release-state/v1';
  sequence: number;
  current: Record<string, unknown> | null;
  previous: Record<string, unknown> | null;
  candidate: Record<string, unknown> | null;
  failures: Record<string, unknown>[];
  revocation: Record<string, unknown> | null;
}

export interface ElectronReleaseStateCoordinator {
  load(): Promise<ElectronReleaseState>;
  installCandidate(pack: ElectronPackReference, decisionDigest: `sha256:${string}`): Promise<ElectronReleaseState>;
  activateCandidate(decision: Record<string, unknown>, customerAuthorizationDigest: `sha256:${string}`): Promise<ElectronReleaseState>;
  rejectCandidate(failureBundleDigest: `sha256:${string}`): Promise<ElectronReleaseState>;
  rollback(customerAuthorizationDigest: `sha256:${string}`): Promise<ElectronReleaseState>;
  applyRevocationSnapshot(snapshot: ElectronRevocationSnapshot): Promise<ElectronReleaseState>;
  resolveCurrent(): Promise<ElectronPackReference>;
}

export declare function validateElectronReleaseState(value: unknown): ElectronReleaseState;
export declare function createElectronReleaseStateCoordinator(options: {
  stateStore: ElectronReleaseStateStore;
  verifyReleaseDecision(decision: Record<string, unknown>): Promise<boolean> | boolean;
  verifyRevocationSnapshot(snapshot: ElectronRevocationSnapshot): Promise<boolean> | boolean;
  now?: () => string;
}): ElectronReleaseStateCoordinator;
