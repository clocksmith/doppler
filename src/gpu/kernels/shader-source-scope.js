import { hashBytesSha256 } from '../../formats/canonical-hash.js';
import { getSharedDeviceEpoch } from '../device-state.js';

const scopes = new WeakMap();
const scopeIds = new WeakMap();
let nextScopeId = 1;
const storageScopes = new WeakMap();
let activeScope = null;
let pending = Promise.resolve();

export function createShaderSourceScope(sources) {
  if (!(sources instanceof Map)) throw new Error('Shader source scope requires a source Map.');
  const copied = new Map();
  for (const [filename, source] of sources) {
    if (typeof filename !== 'string' || !/^[\w.-]+\.wgsl$/.test(filename) || typeof source !== 'string') {
      throw new Error('Shader source scope requires exact WGSL filenames and source text.');
    }
    copied.set(filename, Object.freeze({ source, digest: hashBytesSha256(new TextEncoder().encode(source)) }));
  }
  const scope = Object.freeze({});
  scopes.set(scope, copied);
  scopeIds.set(scope, nextScopeId++);
  return scope;
}

export function bindStorageShaderSourceScope(storage, scope) {
  if (!scopes.has(scope)) throw new Error('Unknown shader source scope.');
  if (storageScopes.has(storage)) throw new Error('Storage shader source scope is already bound.');
  storageScopes.set(storage, scope);
}

export function getStorageShaderSourceScope(storage) {
  return storageScopes.get(storage) ?? null;
}

export function getScopedShaderSource(filename) {
  if (activeScope === null) return null;
  const entry = scopes.get(activeScope).get(filename);
  if (!entry) throw new Error(`Shader ${filename} is outside the verified Capsule source closure.`);
  return entry;
}

export function getShaderScopeCacheKey() {
  return `${getSharedDeviceEpoch()}:${activeScope === null ? 0 : scopeIds.get(activeScope)}`;
}

// The legacy kernels use realm-local caches. Public pipeline calls lease this
// context for their complete invocation, including iterator cleanup. Internal
// calls run on the original pipeline and never acquire a second lease.
async function acquire(scope) {
  if (scope !== null && !scopes.has(scope)) throw new Error('Unknown shader source scope.');
  const previous = pending;
  let release;
  pending = new Promise((resolve) => { release = resolve; });
  await previous;
  activeScope = scope;
  return () => { activeScope = null; release(); };
}

export async function runWithShaderSourceScope(scope, action) {
  const release = await acquire(scope);
  try { return await action(); } finally { release(); }
}

export async function* streamWithShaderSourceScope(scope, action) {
  const release = await acquire(scope);
  try { yield* action(); } finally { release(); }
}
