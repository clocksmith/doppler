function assertPlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`support tiers: ${label} must be an object.`);
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`support tiers: ${label} must be a non-empty string.`);
  }
}

function assertBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new Error(`support tiers: ${label} must be a boolean.`);
  }
}

function assertNullableString(value, label) {
  if (value === null) {
    return;
  }
  assertString(value, label);
}

function assertStringArray(value, label) {
  if (!Array.isArray(value)) {
    throw new Error(`support tiers: ${label} must be an array.`);
  }
  for (let i = 0; i < value.length; i += 1) {
    assertString(value[i], `${label}[${i}]`);
  }
}

function assertEnum(value, allowed, label) {
  assertString(value, label);
  if (!allowed.includes(value)) {
    throw new Error(`support tiers: ${label} must be one of ${allowed.join(', ')}.`);
  }
}

export const SUPPORT_TIER_REGISTRY_SCHEMA_VERSION = 1;

export const SUPPORT_TIERS = Object.freeze(['tier1', 'experimental', 'internal-only']);
export const SUPPORT_SCOPES = Object.freeze([
  'api',
  'cli',
  'demo',
  'format',
  'runtime',
  'integration',
  'browser',
]);
export const CLAIM_VISIBILITY_LEVELS = Object.freeze(['primary', 'secondary', 'none']);

export const DEFAULT_SUPPORT_TIER_REGISTRY = Object.freeze({
  schemaVersion: SUPPORT_TIER_REGISTRY_SCHEMA_VERSION,
  source: 'doppler',
  updatedAtUtc: '1970-01-01T00:00:00.000Z',
  subsystems: [],
});

export function validateSupportTierRegistry(registry) {
  assertPlainObject(registry, 'registry');
  if (registry.schemaVersion !== SUPPORT_TIER_REGISTRY_SCHEMA_VERSION) {
    throw new Error(
      `support tiers: schemaVersion must be ${SUPPORT_TIER_REGISTRY_SCHEMA_VERSION}.`
    );
  }
  if (registry.source !== 'doppler') {
    throw new Error('support tiers: source must be "doppler".');
  }
  assertString(registry.updatedAtUtc, 'updatedAtUtc');
  if (!Array.isArray(registry.subsystems)) {
    throw new Error('support tiers: subsystems must be an array.');
  }

  const seenIds = new Set();
  for (let index = 0; index < registry.subsystems.length; index += 1) {
    const entry = registry.subsystems[index];
    const label = `subsystems[${index}]`;
    assertPlainObject(entry, label);
    assertString(entry.id, `${label}.id`);
    if (seenIds.has(entry.id)) {
      throw new Error(`support tiers: duplicate subsystem id "${entry.id}".`);
    }
    seenIds.add(entry.id);
    assertString(entry.label, `${label}.label`);
    assertEnum(entry.scope, SUPPORT_SCOPES, `${label}.scope`);
    assertEnum(entry.tier, SUPPORT_TIERS, `${label}.tier`);
    assertString(entry.owner, `${label}.owner`);
    assertBoolean(entry.userFacing, `${label}.userFacing`);
    assertBoolean(entry.demoDefault, `${label}.demoDefault`);
    assertBoolean(entry.exported, `${label}.exported`);
    assertEnum(entry.claimVisibility, CLAIM_VISIBILITY_LEVELS, `${label}.claimVisibility`);
    assertNullableString(entry.packageExport, `${label}.packageExport`);
    assertNullableString(entry.command, `${label}.command`);
    assertStringArray(entry.docs, `${label}.docs`);
    assertStringArray(entry.entrypoints, `${label}.entrypoints`);
    assertString(entry.notes, `${label}.notes`);

    if (entry.tier === 'internal-only' && (entry.userFacing || entry.demoDefault || entry.exported)) {
      throw new Error(
        `support tiers: internal-only subsystem "${entry.id}" cannot be userFacing, demoDefault, or exported.`
      );
    }
    if (entry.demoDefault && entry.tier !== 'tier1') {
      throw new Error(`support tiers: demoDefault subsystem "${entry.id}" must be tier1.`);
    }
    if (entry.claimVisibility !== 'none' && !entry.userFacing) {
      throw new Error(
        `support tiers: claimVisibility for "${entry.id}" requires userFacing=true.`
      );
    }
    if (entry.exported && entry.packageExport == null) {
      throw new Error(
        `support tiers: exported subsystem "${entry.id}" must declare packageExport.`
      );
    }
    if (!entry.exported && entry.packageExport != null) {
      throw new Error(
        `support tiers: non-exported subsystem "${entry.id}" must not declare packageExport.`
      );
    }
  }

  return registry;
}
