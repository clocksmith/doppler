export declare const EXECUTION_ROUTING_AUDIT_SCHEMA:
  'doppler.execution-routing-audit/v1';

export interface RegisteredVariantReference {
  operation: string;
  variantId: string;
  descriptorDigest: `sha256:${string}`;
  kernelDigest: `sha256:${string}` | null;
  requires: string[];
}

export interface ExecutionRoutingOpportunity {
  phase: 'preLayer' | 'prefill' | 'decode' | 'postLayer';
  roles: string[];
  kernelId: string;
  reason:
    | 'exact-head-prefill-available'
    | 'tiled-prefill-variant-available'
    | 'f16-output-variant-available';
  selected: RegisteredVariantReference;
  candidate: RegisteredVariantReference;
  disposition: 'calibration-required';
}

export interface ExecutionRoutingAudit {
  schema: 'doppler.execution-routing-audit/v1';
  modelId: string | null;
  artifactDigest: string | null;
  executionGraphDigest: `sha256:${string}`;
  integrity: Array<{
    kernelId: string;
    key: string;
    declaredDigest: string | null;
    expectedDigest: string | null;
    registeredVariants: string[];
    status: 'verified' | 'digest-unregistered' | 'digest-mismatch';
  }>;
  opportunities: ExecutionRoutingOpportunity[];
  digest: `sha256:${string}`;
}

export declare function auditManifestExecutionRouting(
  manifest: Record<string, unknown>,
  registry: Record<string, unknown>,
  kernelDigests: Readonly<Record<string, string>>
): ExecutionRoutingAudit;
