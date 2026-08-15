export interface DopplerResolutionPolicy {
  allowedArtifactVariantIds?: readonly `sha256:${string}`[] | null;
  allowedExecutionIds?: readonly `sha256:${string}`[] | null;
}

export interface ResolvedDopplerResolutionPolicy {
  readonly allowedArtifactVariantIds: NonNullable<DopplerResolutionPolicy['allowedArtifactVariantIds']> | null;
  readonly allowedExecutionIds: NonNullable<DopplerResolutionPolicy['allowedExecutionIds']> | null;
}

export declare function resolveResolutionPolicy(
  value?: DopplerResolutionPolicy | null
): ResolvedDopplerResolutionPolicy;
export declare function assertArtifactVariantAllowed(policy: ResolvedDopplerResolutionPolicy, manifestHash: string): void;
export declare function assertExecutionAllowed(policy: ResolvedDopplerResolutionPolicy, resolvedExecutionId: string): void;
export declare function assertExecutionMayStart(policy: ResolvedDopplerResolutionPolicy): void;
export declare function assertUnreceiptedExecutionAllowed(
  policy: ResolvedDopplerResolutionPolicy,
  apiName: string
): void;
