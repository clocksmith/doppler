export const ECOSYSTEM_STABILITY_MODES = ['semver', 'calendar'];
export const ECOSYSTEM_RANKING_MODES = ['policy', 'relevance', 'blended'];
export const ECOSYSTEM_INCENTIVE_MODES = ['credits', 'reputation', 'hybrid'];
export const ECOSYSTEM_ANTISYBIL_COST = ['none', 'low', 'medium', 'high'];
export const ECOSYSTEM_ENFORCEMENT_MODES = ['progressive', 'strict'];
export const ECOSYSTEM_FAILOVER_TIERS = ['peer', 'relay', 'origin'];
export const ECOSYSTEM_NOTARIZATION_ALGORITHMS = ['ed25519', 'ecdsa-p256', 'rsa-pss-sha256'];

export const DEFAULT_ECOSYSTEM_CONFIG = {
  schemaVersion: 1,
  tenantTrust: {
    enabled: false,
    sso: {
      required: false,
      providers: ['oidc'],
    },
    rbac: {
      enforced: true,
      roleModel: 'org-project',
    },
    namespaceOwnership: {
      enforced: true,
    },
    publishSigning: {
      required: true,
      algorithm: 'ed25519',
    },
  },
  publishWorkflow: {
    enabled: false,
    channels: ['canary', 'stable'],
    gates: {
      schemaValidation: true,
      metadataValidation: true,
      reproducibilityCheck: false,
      malwareScan: false,
    },
    rollback: {
      enabled: true,
      maxRollbackDepth: 20,
    },
  },
  discovery: {
    enabled: false,
    ranking: {
      mode: 'policy',
      signals: {
        qualityEvidenceWeight: 0.35,
        trustWeight: 0.25,
        compatibilityWeight: 0.2,
        adoptionWeight: 0.2,
      },
      minEvidenceRuns: 1,
    },
    badges: {
      enabled: true,
    },
    recommendations: {
      enabled: true,
      dependencyAware: true,
    },
  },
  artifactRegistry: {
    enabled: true,
    types: ['model', 'dataset', 'adapter', 'eval', 'prompt', 'app'],
    linking: {
      enforced: true,
      signedLinks: true,
    },
  },
  hostedAccess: {
    enabled: false,
    apiKeys: {
      enabled: true,
      rotationDays: 90,
    },
    quotas: {
      enabled: true,
      defaultRequestsPerMinute: 600,
      burstMultiplier: 2,
    },
    billing: {
      meteringEnabled: false,
      currency: 'usd',
    },
    endpoints: {
      managedInference: true,
    },
  },
  compliance: {
    enabled: false,
    auditLog: {
      immutable: true,
      retentionDays: 3650,
      legalHold: true,
    },
    jurisdiction: {
      enabled: false,
      defaultRegion: 'global',
    },
    takedown: {
      automated: false,
      slaHours: 24,
    },
    exportControl: {
      enabled: false,
      denyUnknownClassification: true,
    },
  },
  provenance: {
    enabled: true,
    notarization: {
      requiredForPublic: true,
      allowUnsignedPrivate: false,
      algorithm: 'ed25519',
    },
    attestations: {
      required: true,
      formats: ['slsa-v1'],
    },
    revocation: {
      enabled: true,
      propagationSlaMinutes: 15,
    },
  },
  developerEcosystem: {
    enabled: true,
    sdks: {
      js: true,
      python: true,
      agents: true,
    },
    ciHooks: {
      publishVerifyPromote: true,
    },
    migrations: {
      hfImport: true,
      registryAdapters: true,
    },
    contracts: {
      stability: 'semver',
    },
  },
  networkEffects: {
    enabled: false,
    socialSignals: {
      follows: true,
      stars: true,
      trends: true,
    },
    trustScores: {
      enabled: true,
      decayHalfLifeDays: 30,
    },
    incentives: {
      enabled: false,
      mode: 'credits',
    },
  },
  abuse: {
    enabled: true,
    antiSybil: {
      enabled: false,
      minCost: 'none',
    },
    investigation: {
      enabled: true,
      queueSlaHours: 24,
    },
    enforcement: {
      mode: 'progressive',
      allowAppeal: true,
      recoveryWindowDays: 30,
    },
  },
  qualityEvidence: {
    enabled: true,
    registry: {
      requiredForRanking: true,
      immutableHistory: true,
    },
    gating: {
      promotionRequiresEvidence: false,
      minVerifiedRuns: 3,
    },
    verifiability: {
      requireArtifactBinding: true,
      requireSignedRuns: false,
    },
  },
  reliability: {
    enabled: true,
    edgeRelay: {
      policy: 'hybrid',
      minRelayRegions: 2,
    },
    failover: {
      tiers: ['peer', 'relay', 'origin'],
      maxTierSwitches: 3,
    },
    sla: {
      enabled: false,
      targetAvailability: 0.999,
      targetP95Ms: 2000,
    },
  },
};

function mergeSection(base, overrides) {
  return {
    ...base,
    ...(overrides || {}),
  };
}

function mergeArray(base, overrides) {
  if (!Array.isArray(overrides)) {
    return [...base];
  }
  return [...overrides];
}

export function mergeEcosystemConfig(base = DEFAULT_ECOSYSTEM_CONFIG, overrides = {}) {
  const next = mergeSection(base, overrides);
  return {
    ...next,
    tenantTrust: {
      ...mergeSection(base.tenantTrust, overrides.tenantTrust),
      sso: {
        ...mergeSection(base.tenantTrust.sso, overrides.tenantTrust?.sso),
        providers: mergeArray(base.tenantTrust.sso.providers, overrides.tenantTrust?.sso?.providers),
      },
      rbac: mergeSection(base.tenantTrust.rbac, overrides.tenantTrust?.rbac),
      namespaceOwnership: mergeSection(base.tenantTrust.namespaceOwnership, overrides.tenantTrust?.namespaceOwnership),
      publishSigning: mergeSection(base.tenantTrust.publishSigning, overrides.tenantTrust?.publishSigning),
    },
    publishWorkflow: {
      ...mergeSection(base.publishWorkflow, overrides.publishWorkflow),
      channels: mergeArray(base.publishWorkflow.channels, overrides.publishWorkflow?.channels),
      gates: mergeSection(base.publishWorkflow.gates, overrides.publishWorkflow?.gates),
      rollback: mergeSection(base.publishWorkflow.rollback, overrides.publishWorkflow?.rollback),
    },
    discovery: {
      ...mergeSection(base.discovery, overrides.discovery),
      ranking: {
        ...mergeSection(base.discovery.ranking, overrides.discovery?.ranking),
        signals: mergeSection(base.discovery.ranking.signals, overrides.discovery?.ranking?.signals),
      },
      badges: mergeSection(base.discovery.badges, overrides.discovery?.badges),
      recommendations: mergeSection(base.discovery.recommendations, overrides.discovery?.recommendations),
    },
    artifactRegistry: {
      ...mergeSection(base.artifactRegistry, overrides.artifactRegistry),
      types: mergeArray(base.artifactRegistry.types, overrides.artifactRegistry?.types),
      linking: mergeSection(base.artifactRegistry.linking, overrides.artifactRegistry?.linking),
    },
    hostedAccess: {
      ...mergeSection(base.hostedAccess, overrides.hostedAccess),
      apiKeys: mergeSection(base.hostedAccess.apiKeys, overrides.hostedAccess?.apiKeys),
      quotas: mergeSection(base.hostedAccess.quotas, overrides.hostedAccess?.quotas),
      billing: mergeSection(base.hostedAccess.billing, overrides.hostedAccess?.billing),
      endpoints: mergeSection(base.hostedAccess.endpoints, overrides.hostedAccess?.endpoints),
    },
    compliance: {
      ...mergeSection(base.compliance, overrides.compliance),
      auditLog: mergeSection(base.compliance.auditLog, overrides.compliance?.auditLog),
      jurisdiction: mergeSection(base.compliance.jurisdiction, overrides.compliance?.jurisdiction),
      takedown: mergeSection(base.compliance.takedown, overrides.compliance?.takedown),
      exportControl: mergeSection(base.compliance.exportControl, overrides.compliance?.exportControl),
    },
    provenance: {
      ...mergeSection(base.provenance, overrides.provenance),
      notarization: mergeSection(base.provenance.notarization, overrides.provenance?.notarization),
      attestations: {
        ...mergeSection(base.provenance.attestations, overrides.provenance?.attestations),
        formats: mergeArray(base.provenance.attestations.formats, overrides.provenance?.attestations?.formats),
      },
      revocation: mergeSection(base.provenance.revocation, overrides.provenance?.revocation),
    },
    developerEcosystem: {
      ...mergeSection(base.developerEcosystem, overrides.developerEcosystem),
      sdks: mergeSection(base.developerEcosystem.sdks, overrides.developerEcosystem?.sdks),
      ciHooks: mergeSection(base.developerEcosystem.ciHooks, overrides.developerEcosystem?.ciHooks),
      migrations: mergeSection(base.developerEcosystem.migrations, overrides.developerEcosystem?.migrations),
      contracts: mergeSection(base.developerEcosystem.contracts, overrides.developerEcosystem?.contracts),
    },
    networkEffects: {
      ...mergeSection(base.networkEffects, overrides.networkEffects),
      socialSignals: mergeSection(base.networkEffects.socialSignals, overrides.networkEffects?.socialSignals),
      trustScores: mergeSection(base.networkEffects.trustScores, overrides.networkEffects?.trustScores),
      incentives: mergeSection(base.networkEffects.incentives, overrides.networkEffects?.incentives),
    },
    abuse: {
      ...mergeSection(base.abuse, overrides.abuse),
      antiSybil: mergeSection(base.abuse.antiSybil, overrides.abuse?.antiSybil),
      investigation: mergeSection(base.abuse.investigation, overrides.abuse?.investigation),
      enforcement: mergeSection(base.abuse.enforcement, overrides.abuse?.enforcement),
    },
    qualityEvidence: {
      ...mergeSection(base.qualityEvidence, overrides.qualityEvidence),
      registry: mergeSection(base.qualityEvidence.registry, overrides.qualityEvidence?.registry),
      gating: mergeSection(base.qualityEvidence.gating, overrides.qualityEvidence?.gating),
      verifiability: mergeSection(base.qualityEvidence.verifiability, overrides.qualityEvidence?.verifiability),
    },
    reliability: {
      ...mergeSection(base.reliability, overrides.reliability),
      edgeRelay: mergeSection(base.reliability.edgeRelay, overrides.reliability?.edgeRelay),
      failover: {
        ...mergeSection(base.reliability.failover, overrides.reliability?.failover),
        tiers: mergeArray(base.reliability.failover.tiers, overrides.reliability?.failover?.tiers),
      },
      sla: mergeSection(base.reliability.sla, overrides.reliability?.sla),
    },
  };
}

export function createEcosystemConfig(overrides = {}) {
  return mergeEcosystemConfig(DEFAULT_ECOSYSTEM_CONFIG, overrides);
}

function assertObject(label, value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`DopplerConfigError: ${label} must be an object.`);
  }
}

function assertBoolean(label, value) {
  if (value !== true && value !== false) {
    throw new Error(`DopplerConfigError: ${label} must be boolean.`);
  }
}

function assertPositiveInt(label, value) {
  if (!Number.isFinite(value) || Math.floor(value) !== value || value <= 0) {
    throw new Error(`DopplerConfigError: ${label} must be a positive integer.`);
  }
}

function assertNumberInRange(label, value, min, max) {
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new Error(`DopplerConfigError: ${label} must be in range [${min}, ${max}].`);
  }
}

function assertStringIn(label, value, allowed) {
  if (typeof value !== 'string' || !allowed.includes(value)) {
    throw new Error(`DopplerConfigError: ${label} must be one of ${allowed.join(', ')}.`);
  }
}

function assertArrayOfNonEmptyStrings(label, value) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`DopplerConfigError: ${label} must be a non-empty string array.`);
  }
  for (let i = 0; i < value.length; i += 1) {
    const entry = value[i];
    if (typeof entry !== 'string' || !entry.trim()) {
      throw new Error(`DopplerConfigError: ${label}[${i}] must be a non-empty string.`);
    }
  }
}

export function validateEcosystemConfig(config) {
  assertObject('runtime.shared.ecosystem', config);
  if (config.schemaVersion !== 1) {
    throw new Error('DopplerConfigError: runtime.shared.ecosystem.schemaVersion must be 1.');
  }

  assertObject('runtime.shared.ecosystem.tenantTrust', config.tenantTrust);
  assertBoolean('runtime.shared.ecosystem.tenantTrust.enabled', config.tenantTrust.enabled);
  assertArrayOfNonEmptyStrings('runtime.shared.ecosystem.tenantTrust.sso.providers', config.tenantTrust.sso.providers);
  assertBoolean('runtime.shared.ecosystem.tenantTrust.publishSigning.required', config.tenantTrust.publishSigning.required);
  assertStringIn(
    'runtime.shared.ecosystem.tenantTrust.publishSigning.algorithm',
    config.tenantTrust.publishSigning.algorithm,
    ECOSYSTEM_NOTARIZATION_ALGORITHMS
  );

  assertObject('runtime.shared.ecosystem.publishWorkflow', config.publishWorkflow);
  assertBoolean('runtime.shared.ecosystem.publishWorkflow.enabled', config.publishWorkflow.enabled);
  assertArrayOfNonEmptyStrings('runtime.shared.ecosystem.publishWorkflow.channels', config.publishWorkflow.channels);
  assertPositiveInt(
    'runtime.shared.ecosystem.publishWorkflow.rollback.maxRollbackDepth',
    config.publishWorkflow.rollback.maxRollbackDepth
  );

  assertObject('runtime.shared.ecosystem.discovery', config.discovery);
  assertBoolean('runtime.shared.ecosystem.discovery.enabled', config.discovery.enabled);
  assertStringIn(
    'runtime.shared.ecosystem.discovery.ranking.mode',
    config.discovery.ranking.mode,
    ECOSYSTEM_RANKING_MODES
  );
  assertPositiveInt(
    'runtime.shared.ecosystem.discovery.ranking.minEvidenceRuns',
    config.discovery.ranking.minEvidenceRuns
  );
  const rankingSignals = config.discovery.ranking.signals;
  const weightSum = Number(rankingSignals.qualityEvidenceWeight)
    + Number(rankingSignals.trustWeight)
    + Number(rankingSignals.compatibilityWeight)
    + Number(rankingSignals.adoptionWeight);
  assertNumberInRange(
    'runtime.shared.ecosystem.discovery.ranking.signals sum',
    weightSum,
    0.999,
    1.001
  );

  assertObject('runtime.shared.ecosystem.artifactRegistry', config.artifactRegistry);
  assertBoolean('runtime.shared.ecosystem.artifactRegistry.enabled', config.artifactRegistry.enabled);
  assertArrayOfNonEmptyStrings('runtime.shared.ecosystem.artifactRegistry.types', config.artifactRegistry.types);

  assertObject('runtime.shared.ecosystem.hostedAccess', config.hostedAccess);
  assertBoolean('runtime.shared.ecosystem.hostedAccess.enabled', config.hostedAccess.enabled);
  assertPositiveInt(
    'runtime.shared.ecosystem.hostedAccess.apiKeys.rotationDays',
    config.hostedAccess.apiKeys.rotationDays
  );
  assertPositiveInt(
    'runtime.shared.ecosystem.hostedAccess.quotas.defaultRequestsPerMinute',
    config.hostedAccess.quotas.defaultRequestsPerMinute
  );

  assertObject('runtime.shared.ecosystem.compliance', config.compliance);
  assertBoolean('runtime.shared.ecosystem.compliance.enabled', config.compliance.enabled);
  assertPositiveInt(
    'runtime.shared.ecosystem.compliance.auditLog.retentionDays',
    config.compliance.auditLog.retentionDays
  );
  assertPositiveInt(
    'runtime.shared.ecosystem.compliance.takedown.slaHours',
    config.compliance.takedown.slaHours
  );

  assertObject('runtime.shared.ecosystem.provenance', config.provenance);
  assertBoolean('runtime.shared.ecosystem.provenance.enabled', config.provenance.enabled);
  assertStringIn(
    'runtime.shared.ecosystem.provenance.notarization.algorithm',
    config.provenance.notarization.algorithm,
    ECOSYSTEM_NOTARIZATION_ALGORITHMS
  );
  assertArrayOfNonEmptyStrings(
    'runtime.shared.ecosystem.provenance.attestations.formats',
    config.provenance.attestations.formats
  );
  assertPositiveInt(
    'runtime.shared.ecosystem.provenance.revocation.propagationSlaMinutes',
    config.provenance.revocation.propagationSlaMinutes
  );

  assertObject('runtime.shared.ecosystem.developerEcosystem', config.developerEcosystem);
  assertBoolean('runtime.shared.ecosystem.developerEcosystem.enabled', config.developerEcosystem.enabled);
  assertStringIn(
    'runtime.shared.ecosystem.developerEcosystem.contracts.stability',
    config.developerEcosystem.contracts.stability,
    ECOSYSTEM_STABILITY_MODES
  );

  assertObject('runtime.shared.ecosystem.networkEffects', config.networkEffects);
  assertBoolean('runtime.shared.ecosystem.networkEffects.enabled', config.networkEffects.enabled);
  assertStringIn(
    'runtime.shared.ecosystem.networkEffects.incentives.mode',
    config.networkEffects.incentives.mode,
    ECOSYSTEM_INCENTIVE_MODES
  );
  assertPositiveInt(
    'runtime.shared.ecosystem.networkEffects.trustScores.decayHalfLifeDays',
    config.networkEffects.trustScores.decayHalfLifeDays
  );

  assertObject('runtime.shared.ecosystem.abuse', config.abuse);
  assertBoolean('runtime.shared.ecosystem.abuse.enabled', config.abuse.enabled);
  assertStringIn(
    'runtime.shared.ecosystem.abuse.antiSybil.minCost',
    config.abuse.antiSybil.minCost,
    ECOSYSTEM_ANTISYBIL_COST
  );
  assertStringIn(
    'runtime.shared.ecosystem.abuse.enforcement.mode',
    config.abuse.enforcement.mode,
    ECOSYSTEM_ENFORCEMENT_MODES
  );
  assertPositiveInt(
    'runtime.shared.ecosystem.abuse.enforcement.recoveryWindowDays',
    config.abuse.enforcement.recoveryWindowDays
  );

  assertObject('runtime.shared.ecosystem.qualityEvidence', config.qualityEvidence);
  assertBoolean('runtime.shared.ecosystem.qualityEvidence.enabled', config.qualityEvidence.enabled);
  assertPositiveInt(
    'runtime.shared.ecosystem.qualityEvidence.gating.minVerifiedRuns',
    config.qualityEvidence.gating.minVerifiedRuns
  );

  assertObject('runtime.shared.ecosystem.reliability', config.reliability);
  assertBoolean('runtime.shared.ecosystem.reliability.enabled', config.reliability.enabled);
  assertPositiveInt(
    'runtime.shared.ecosystem.reliability.edgeRelay.minRelayRegions',
    config.reliability.edgeRelay.minRelayRegions
  );
  assertPositiveInt(
    'runtime.shared.ecosystem.reliability.failover.maxTierSwitches',
    config.reliability.failover.maxTierSwitches
  );
  assertArrayOfNonEmptyStrings(
    'runtime.shared.ecosystem.reliability.failover.tiers',
    config.reliability.failover.tiers
  );
  for (const tier of config.reliability.failover.tiers) {
    if (!ECOSYSTEM_FAILOVER_TIERS.includes(tier)) {
      throw new Error(
        `DopplerConfigError: runtime.shared.ecosystem.reliability.failover.tiers entries must be ${ECOSYSTEM_FAILOVER_TIERS.join(', ')}.`
      );
    }
  }
  if (!config.reliability.failover.tiers.includes('origin')) {
    throw new Error('DopplerConfigError: runtime.shared.ecosystem.reliability.failover.tiers must include "origin".');
  }
  const uniqueTiers = new Set(config.reliability.failover.tiers);
  if (uniqueTiers.size !== config.reliability.failover.tiers.length) {
    throw new Error('DopplerConfigError: runtime.shared.ecosystem.reliability.failover.tiers must be unique.');
  }
  assertNumberInRange(
    'runtime.shared.ecosystem.reliability.sla.targetAvailability',
    config.reliability.sla.targetAvailability,
    0.9,
    1
  );
  assertPositiveInt(
    'runtime.shared.ecosystem.reliability.sla.targetP95Ms',
    config.reliability.sla.targetP95Ms
  );
}
