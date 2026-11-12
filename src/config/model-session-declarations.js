import declarations from './runtime/model-sessions.json' with { type: 'json' };

// Per-model runtime session declarations for published manifests that do not
// declare inference.session.kvcache.maxSeqLen. The file is data keyed by exact
// modelId; nothing here derives a value, and an unlisted model still fails fast
// at KV allocation.
export const MODEL_SESSION_DECLARATIONS_SCHEMA_ID = 'doppler.model-session-declarations/v1';
export const MODEL_SESSION_DECLARATIONS_PATH = 'src/config/runtime/model-sessions.json';

/** @param {unknown} value @returns {value is number} */
function isPositiveSafeInteger(value) {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
}

/**
 * @param {any} document
 * @returns {Map<string, import('./model-session-declarations.js').ModelSessionDeclaration>}
 */
function validateDeclarations(document) {
  if (document?.$schema !== MODEL_SESSION_DECLARATIONS_SCHEMA_ID) {
    throw new Error(
      `${MODEL_SESSION_DECLARATIONS_PATH} schema mismatch: expected "${MODEL_SESSION_DECLARATIONS_SCHEMA_ID}", ` +
      `got "${document?.$schema}".`
    );
  }
  const models = document.models;
  if (!models || typeof models !== 'object' || Array.isArray(models)) {
    throw new Error(`${MODEL_SESSION_DECLARATIONS_PATH} models must be an object keyed by modelId.`);
  }
  const entries = new Map();
  for (const [modelId, entry] of Object.entries(models)) {
    const { contextLength, maxSeqLen } = entry ?? {};
    if (!isPositiveSafeInteger(contextLength) || !isPositiveSafeInteger(maxSeqLen) || maxSeqLen > contextLength) {
      throw new Error(
        `${MODEL_SESSION_DECLARATIONS_PATH} "${modelId}" needs positive integer contextLength and maxSeqLen ` +
        `with maxSeqLen <= contextLength; got ${JSON.stringify(entry)}.`
      );
    }
    entries.set(modelId, Object.freeze({ contextLength, maxSeqLen }));
  }
  return entries;
}

const DECLARED_MODEL_SESSIONS = validateDeclarations(declarations);

/** @type {import('./model-session-declarations.js').getModelSessionDeclaration} */
export function getModelSessionDeclaration(modelId) {
  return DECLARED_MODEL_SESSIONS.get(modelId) ?? null;
}

export function listModelSessionDeclarations() {
  return Object.freeze([...DECLARED_MODEL_SESSIONS.keys()]);
}

// Returns the manifest session with the declared KV length filled in when the
// manifest itself declares none. A manifest declaration is model-owned and is
// returned unchanged. The declaration binds to the manifest context it was
// computed from; a different context fails instead of applying a stale length.
/**
 * @param {any} manifest
 * @returns {Record<string, any>}
 */
export function applyModelSessionDeclaration(manifest) {
  const session = manifest?.inference?.session ?? {};
  if (isPositiveSafeInteger(session?.kvcache?.maxSeqLen)) {
    return session;
  }
  const modelId = typeof manifest?.modelId === 'string' ? manifest.modelId : null;
  const declaration = modelId ? getModelSessionDeclaration(modelId) : null;
  if (!declaration) {
    return session;
  }
  const contextLength = manifest?.architecture?.maxSeqLen;
  if (contextLength !== declaration.contextLength) {
    throw new Error(
      `${MODEL_SESSION_DECLARATIONS_PATH} declares "${modelId}" for contextLength ${declaration.contextLength}, ` +
      `but the manifest architecture.maxSeqLen is ${String(contextLength)}. Declare ` +
      'inference.session.kvcache.maxSeqLen in the manifest or a runtime profile/override for this artifact.'
    );
  }
  return {
    ...session,
    kvcache: { ...(session.kvcache ?? {}), maxSeqLen: declaration.maxSeqLen },
  };
}
