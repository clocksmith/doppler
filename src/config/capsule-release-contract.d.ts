export declare const CAPSULE_RELEASE_SCHEMA_ID: 'doppler.capsule-release/v1';
export declare const CAPSULE_STATE_SNAPSHOT_SCHEMA_ID: 'doppler.capsule-state-snapshot/v1';

export interface CapsuleReleaseIdentity {
  id: string;
  digest: `sha256:${string}`;
}

export interface CapsuleReleaseContract {
  schema: 'doppler.capsule-release/v1';
  source: {
    repository: string;
    revision: string;
    revisionDigest: `sha256:${string}`;
    provenanceDigest: `sha256:${string}`;
    license: {
      spdxId: string;
      name: string;
      sourceUrl: string;
      textDigest: `sha256:${string}`;
    };
  };
  application: {
    applicationId: string;
    applicationRevision: string;
    applicationRevisionDigest: `sha256:${string}`;
    workload: CapsuleReleaseIdentity;
    oracle: CapsuleReleaseIdentity;
  };
  exclusions: {
    rejectionTypes: string[];
    known: Array<{
      code: string;
      scope: string;
      reason: string;
      evidenceDigest: `sha256:${string}`;
    }>;
  };
  lifecycle: {
    releaseVersion: string;
    supersedes: null | { capsuleId: string; semanticRoot: `sha256:${string}` };
    migration: null | { id: string; policyDigest: `sha256:${string}`; required: boolean };
    failedUpgrade: {
      preservePrevious: true;
      previousCapsuleId: string | null;
      previousSemanticRoot: `sha256:${string}` | null;
    };
  };
  revocation: {
    authorityId: string;
    policyDigest: `sha256:${string}`;
    offlineExpirySeconds: number;
    failClosedAfterExpiry: true;
  };
  stateSnapshot: {
    schema: 'doppler.capsule-state-snapshot/v1';
    format: string;
    identityDigest: `sha256:${string}`;
    portableAcrossTargetIds: string[];
  };
}

export declare function validateCapsuleReleaseContract(
  release: unknown,
  options?: { targetIds?: string[] }
): { ok: boolean; errors: string[] };
