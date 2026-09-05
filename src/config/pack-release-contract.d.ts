export declare const PACK_RELEASE_SCHEMA_ID: 'doppler.pack-release/v1';
export declare const PACK_STATE_SNAPSHOT_SCHEMA_ID: 'doppler.pack-state-snapshot/v1';

export interface PackReleaseIdentity {
  id: string;
  digest: `sha256:${string}`;
}

export interface PackReleaseContract {
  schema: 'doppler.pack-release/v1';
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
    workload: PackReleaseIdentity;
    oracle: PackReleaseIdentity;
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
    supersedes: null | { packId: string; semanticRoot: `sha256:${string}` };
    migration: null | { id: string; policyDigest: `sha256:${string}`; required: boolean };
    failedUpgrade: {
      preservePrevious: true;
      previousPackId: string | null;
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
    schema: 'doppler.pack-state-snapshot/v1';
    format: string;
    identityDigest: `sha256:${string}`;
    portableAcrossTargetIds: string[];
  };
}

export declare function validatePackReleaseContract(
  release: unknown,
  options?: { targetIds?: string[] }
): { ok: boolean; errors: string[] };
