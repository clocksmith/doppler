import { TOOLING_INTENTS } from './tooling.schema.js';

const RUNTIME_PROFILE_STABILITIES = Object.freeze([
  'canonical',
  'experimental',
  'deprecated',
]);

function assertNonEmptyString(value, label) {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`DopplerConfigError: ${label} must be a non-empty string.`);
  }
  return value.trim();
}

function normalizeCompatibleIntents(value, label) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`DopplerConfigError: ${label} must be a non-empty array.`);
  }
  const compatibleIntents = value.map((intent, index) => {
    const normalized = assertNonEmptyString(intent, `${label}[${index}]`);
    if (!TOOLING_INTENTS.includes(normalized)) {
      throw new Error(
        `DopplerConfigError: ${label}[${index}] must be one of ${TOOLING_INTENTS.join(', ')}.`
      );
    }
    return normalized;
  });
  if (new Set(compatibleIntents).size !== compatibleIntents.length) {
    throw new Error(`DopplerConfigError: ${label} must not contain duplicates.`);
  }
  return compatibleIntents;
}

export function validateRuntimeProfileMetadata(profile, label = 'runtime profile') {
  if (!profile || typeof profile !== 'object' || Array.isArray(profile)) {
    throw new Error(`DopplerConfigError: ${label} must be an object.`);
  }
  const id = assertNonEmptyString(profile.id, `${label}.id`);
  const name = assertNonEmptyString(profile.name, `${label}.name`);
  const intent = assertNonEmptyString(profile.intent, `${label}.intent`);
  if (!TOOLING_INTENTS.includes(intent)) {
    throw new Error(
      `DopplerConfigError: ${label}.intent must be one of ${TOOLING_INTENTS.join(', ')}.`
    );
  }
  const compatibleIntents = normalizeCompatibleIntents(
    profile.compatibleIntents,
    `${label}.compatibleIntents`
  );
  if (!compatibleIntents.includes(intent)) {
    throw new Error(
      `DopplerConfigError: ${label}.compatibleIntents must include the profile intent "${intent}".`
    );
  }
  const stability = assertNonEmptyString(profile.stability, `${label}.stability`);
  if (!RUNTIME_PROFILE_STABILITIES.includes(stability)) {
    throw new Error(
      `DopplerConfigError: ${label}.stability must be one of ${RUNTIME_PROFILE_STABILITIES.join(', ')}.`
    );
  }
  const owner = assertNonEmptyString(profile.owner, `${label}.owner`);
  const createdAtUtc = assertNonEmptyString(profile.createdAtUtc, `${label}.createdAtUtc`);
  if (Number.isNaN(Date.parse(createdAtUtc))) {
    throw new Error(`DopplerConfigError: ${label}.createdAtUtc must be an ISO-8601 timestamp.`);
  }
  return Object.freeze({
    id,
    name,
    intent,
    compatibleIntents: Object.freeze([...compatibleIntents]),
    stability,
    owner,
    createdAtUtc,
  });
}

export function assertRuntimeProfileIntentCompatibility(profile, requestIntent, label = 'runtime profile') {
  const metadata = validateRuntimeProfileMetadata(profile, label);
  if (requestIntent === null || requestIntent === undefined) {
    throw new Error(
      `DopplerConfigError: ${label} "${metadata.id}" requires a harness command intent.`
    );
  }
  if (!metadata.compatibleIntents.includes(requestIntent)) {
    throw new Error(
      `DopplerConfigError: ${label} "${metadata.id}" is not compatible with intent "${requestIntent}". ` +
      `Compatible intents: ${metadata.compatibleIntents.join(', ')}.`
    );
  }
  return metadata;
}
