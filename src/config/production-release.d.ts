export type ProductionReleaseEvidenceClass =
  | 'reference-fixture'
  | 'external-candidate'
  | 'external-production';

export interface ProductionReleaseV1 {
  schema: 'doppler.production-release/v1';
  schemaVersion: 1;
  releaseId: string;
  createdAtUtc: string;
  evidenceClass: ProductionReleaseEvidenceClass;
  candidate: Record<string, unknown> & {
    logicalModelId: string;
    sourceRevision: string;
    sourceRevisionDigest: `sha256:${string}`;
    packPath: string;
    packSemanticRoot: `sha256:${string}`;
  };
  application: Record<string, unknown> & {
    applicationId: string;
    platform: 'electron';
    revision: string;
    revisionDigest: `sha256:${string}`;
    rendererEntry: string;
    mainEntry: string;
  };
  acceptance: Record<string, unknown>;
  supportedDevices: Record<string, unknown>;
  previousRelease: Record<string, unknown>;
  rollout: Record<string, unknown> & {
    activationAuthority: 'customer';
    selfPromotionAllowed: false;
  };
  rollback: Record<string, unknown> & { authority: 'customer' };
  revocation: Record<string, unknown>;
  dataCustody: Record<string, unknown>;
  claimBoundary: {
    externalCustomer: boolean;
    commercialClaimAllowed: boolean;
  };
}

export const PRODUCTION_RELEASE_SCHEMA_ID: 'doppler.production-release/v1';
export const PRODUCTION_RELEASE_SCHEMA_VERSION: 1;
export declare function getProductionReleaseSemanticPayload(release: ProductionReleaseV1): Record<string, unknown>;
export declare function hashProductionRelease(release: ProductionReleaseV1): `sha256:${string}`;
export declare function validateProductionRelease(release: unknown): { ok: boolean; errors: string[] };
export declare function assertProductionRelease(release: unknown): ProductionReleaseV1;
