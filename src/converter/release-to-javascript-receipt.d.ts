export declare const RELEASE_TO_JAVASCRIPT_RECEIPT_SCHEMA_ID: 'doppler.release-to-javascript-receipt/v2';

export type SourcePublicationTimestampDisposition = 'observed' | 'unresolved';

export interface ReleaseToJavaScriptReceipt {
  schema: typeof RELEASE_TO_JAVASCRIPT_RECEIPT_SCHEMA_ID;
  campaignId: string;
  source: {
    checkpointId: string;
    revision: string;
    publicationTimestampDisposition: SourcePublicationTimestampDisposition;
    publishedAt: string | null;
  };
  startedAt: string;
  completedAt: string;
  elapsed: { publicationToSignedPackMs: number | null; forgeCampaignMs: number };
  humanInterventions: Array<{ id: string; kind: string; actor: string; disposition: string }>;
  humanAuthoredSemanticDecisions: number;
  unresolvedFacts: unknown[];
  candidates: { generated: number; rejected: unknown[]; accepted: unknown[] };
  acceptedCode: {
    revision: string;
    files: Array<{ path: string; digest: `sha256:${string}` }>;
    digest: `sha256:${string}`;
  };
  qualification: { status: 'passed'; packId: string; packDigest: `sha256:${string}` };
  evidence: Array<{ kind: string; path: string; digest: `sha256:${string}` }>;
  receiptDigest: `sha256:${string}`;
}

export declare function validateReleaseToJavaScriptReceipt(value: unknown): { ok: boolean; errors: string[] };
export declare function createReleaseToJavaScriptReceipt(fields: Omit<
  ReleaseToJavaScriptReceipt,
  'schema' | 'elapsed' | 'humanAuthoredSemanticDecisions' | 'receiptDigest'
>): Readonly<ReleaseToJavaScriptReceipt>;
