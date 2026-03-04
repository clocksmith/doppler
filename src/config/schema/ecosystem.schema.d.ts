export type EcosystemStabilityMode = 'semver' | 'calendar';
export type EcosystemRankingMode = 'policy' | 'relevance' | 'blended';
export type EcosystemIncentiveMode = 'credits' | 'reputation' | 'hybrid';
export type EcosystemAntiSybilCost = 'none' | 'low' | 'medium' | 'high';
export type EcosystemEnforcementMode = 'progressive' | 'strict';
export type EcosystemFailoverTier = 'peer' | 'relay' | 'origin';
export type EcosystemNotarizationAlgorithm = 'ed25519' | 'ecdsa-p256' | 'rsa-pss-sha256';

export interface EcosystemTenantTrustSchema {
  enabled: boolean;
  sso: {
    required: boolean;
    providers: string[];
  };
  rbac: {
    enforced: boolean;
    roleModel: string;
  };
  namespaceOwnership: {
    enforced: boolean;
  };
  publishSigning: {
    required: boolean;
    algorithm: EcosystemNotarizationAlgorithm;
  };
}

export interface EcosystemPublishWorkflowSchema {
  enabled: boolean;
  channels: string[];
  gates: {
    schemaValidation: boolean;
    metadataValidation: boolean;
    reproducibilityCheck: boolean;
    malwareScan: boolean;
  };
  rollback: {
    enabled: boolean;
    maxRollbackDepth: number;
  };
}

export interface EcosystemDiscoverySchema {
  enabled: boolean;
  ranking: {
    mode: EcosystemRankingMode;
    signals: {
      qualityEvidenceWeight: number;
      trustWeight: number;
      compatibilityWeight: number;
      adoptionWeight: number;
    };
    minEvidenceRuns: number;
  };
  badges: {
    enabled: boolean;
  };
  recommendations: {
    enabled: boolean;
    dependencyAware: boolean;
  };
}

export interface EcosystemArtifactRegistrySchema {
  enabled: boolean;
  types: string[];
  linking: {
    enforced: boolean;
    signedLinks: boolean;
  };
}

export interface EcosystemHostedAccessSchema {
  enabled: boolean;
  apiKeys: {
    enabled: boolean;
    rotationDays: number;
  };
  quotas: {
    enabled: boolean;
    defaultRequestsPerMinute: number;
    burstMultiplier: number;
  };
  billing: {
    meteringEnabled: boolean;
    currency: string;
  };
  endpoints: {
    managedInference: boolean;
  };
}

export interface EcosystemComplianceSchema {
  enabled: boolean;
  auditLog: {
    immutable: boolean;
    retentionDays: number;
    legalHold: boolean;
  };
  jurisdiction: {
    enabled: boolean;
    defaultRegion: string;
  };
  takedown: {
    automated: boolean;
    slaHours: number;
  };
  exportControl: {
    enabled: boolean;
    denyUnknownClassification: boolean;
  };
}

export interface EcosystemProvenanceSchema {
  enabled: boolean;
  notarization: {
    requiredForPublic: boolean;
    allowUnsignedPrivate: boolean;
    algorithm: EcosystemNotarizationAlgorithm;
  };
  attestations: {
    required: boolean;
    formats: string[];
  };
  revocation: {
    enabled: boolean;
    propagationSlaMinutes: number;
  };
}

export interface EcosystemDeveloperSchema {
  enabled: boolean;
  sdks: {
    js: boolean;
    python: boolean;
    agents: boolean;
  };
  ciHooks: {
    publishVerifyPromote: boolean;
  };
  migrations: {
    hfImport: boolean;
    registryAdapters: boolean;
  };
  contracts: {
    stability: EcosystemStabilityMode;
  };
}

export interface EcosystemNetworkEffectsSchema {
  enabled: boolean;
  socialSignals: {
    follows: boolean;
    stars: boolean;
    trends: boolean;
  };
  trustScores: {
    enabled: boolean;
    decayHalfLifeDays: number;
  };
  incentives: {
    enabled: boolean;
    mode: EcosystemIncentiveMode;
  };
}

export interface EcosystemAbuseSchema {
  enabled: boolean;
  antiSybil: {
    enabled: boolean;
    minCost: EcosystemAntiSybilCost;
  };
  investigation: {
    enabled: boolean;
    queueSlaHours: number;
  };
  enforcement: {
    mode: EcosystemEnforcementMode;
    allowAppeal: boolean;
    recoveryWindowDays: number;
  };
}

export interface EcosystemQualityEvidenceSchema {
  enabled: boolean;
  registry: {
    requiredForRanking: boolean;
    immutableHistory: boolean;
  };
  gating: {
    promotionRequiresEvidence: boolean;
    minVerifiedRuns: number;
  };
  verifiability: {
    requireArtifactBinding: boolean;
    requireSignedRuns: boolean;
  };
}

export interface EcosystemReliabilitySchema {
  enabled: boolean;
  edgeRelay: {
    policy: string;
    minRelayRegions: number;
  };
  failover: {
    tiers: EcosystemFailoverTier[];
    maxTierSwitches: number;
  };
  sla: {
    enabled: boolean;
    targetAvailability: number;
    targetP95Ms: number;
  };
}

export interface EcosystemConfigSchema {
  schemaVersion: 1;
  tenantTrust: EcosystemTenantTrustSchema;
  publishWorkflow: EcosystemPublishWorkflowSchema;
  discovery: EcosystemDiscoverySchema;
  artifactRegistry: EcosystemArtifactRegistrySchema;
  hostedAccess: EcosystemHostedAccessSchema;
  compliance: EcosystemComplianceSchema;
  provenance: EcosystemProvenanceSchema;
  developerEcosystem: EcosystemDeveloperSchema;
  networkEffects: EcosystemNetworkEffectsSchema;
  abuse: EcosystemAbuseSchema;
  qualityEvidence: EcosystemQualityEvidenceSchema;
  reliability: EcosystemReliabilitySchema;
}

export type EcosystemConfigOverrides = Partial<{
  [K in keyof EcosystemConfigSchema]:
    EcosystemConfigSchema[K] extends object ? Partial<EcosystemConfigSchema[K]> : EcosystemConfigSchema[K];
}>;

export declare const ECOSYSTEM_STABILITY_MODES: readonly EcosystemStabilityMode[];
export declare const ECOSYSTEM_RANKING_MODES: readonly EcosystemRankingMode[];
export declare const ECOSYSTEM_INCENTIVE_MODES: readonly EcosystemIncentiveMode[];
export declare const ECOSYSTEM_ANTISYBIL_COST: readonly EcosystemAntiSybilCost[];
export declare const ECOSYSTEM_ENFORCEMENT_MODES: readonly EcosystemEnforcementMode[];
export declare const ECOSYSTEM_FAILOVER_TIERS: readonly EcosystemFailoverTier[];
export declare const ECOSYSTEM_NOTARIZATION_ALGORITHMS: readonly EcosystemNotarizationAlgorithm[];

export declare const DEFAULT_ECOSYSTEM_CONFIG: EcosystemConfigSchema;

export declare function mergeEcosystemConfig(
  base?: EcosystemConfigSchema,
  overrides?: EcosystemConfigOverrides
): EcosystemConfigSchema;

export declare function createEcosystemConfig(overrides?: EcosystemConfigOverrides): EcosystemConfigSchema;

export declare function validateEcosystemConfig(config: EcosystemConfigSchema): void;
