/// <reference lib="es2024.promise" />
import { hashBytesSha256 } from '../../formats/canonical-hash.js';
import { getSharedDeviceEpoch } from '../device-state.js';
import { getRequiredWgslFeatures } from '../../config/wgsl-language-contract.js';

/** @typedef {import('./shader-source-scope.js').ShaderSourceScope} ShaderSourceScope */
/** @type {WeakMap<ShaderSourceScope, Map<string, import('./shader-source-scope.js').ShaderSourceEntry>>} */
const scopes = new WeakMap();
/** @type {WeakMap<ShaderSourceScope, number>} */
const scopeIds = new WeakMap();
let nextScopeId = 1;
/** @type {WeakMap<object, ShaderSourceScope>} */
const storageScopes = new WeakMap();
/** @type {ShaderSourceScope | null} */
let activeScope = null;
let pending = Promise.resolve();

/** @type {typeof import('./shader-source-scope.js').createShaderSourceScope} */
export function createShaderSourceScope(sources) {
  if (!(sources instanceof Map)) throw new Error('Shader source scope requires a source Map.');
  const copied = new Map();
  for (const [filename, source] of sources) {
    if (typeof filename !== 'string' || !/^[\w.-]+\.wgsl$/.test(filename) || typeof source !== 'string') {
      throw new Error('Shader source scope requires exact WGSL filenames and source text.');
    }
    copied.set(filename, Object.freeze({ source, digest: hashBytesSha256(new TextEncoder().encode(source)),
      requiredWgslFeatures: Object.freeze(getRequiredWgslFeatures(source)) }));
  }
  const scope = Object.freeze({});
  scopes.set(scope, copied);
  scopeIds.set(scope, nextScopeId++);
  return scope;
}

/** @type {typeof import('./shader-source-scope.js').bindStorageShaderSourceScope} */
export function bindStorageShaderSourceScope(storage, scope) {
  if (!scopes.has(scope)) throw new Error('Unknown shader source scope.');
  if (storageScopes.has(storage)) throw new Error('Storage shader source scope is already bound.');
  storageScopes.set(storage, scope);
}

/** @type {typeof import('./shader-source-scope.js').getStorageShaderSourceScope} */
export function getStorageShaderSourceScope(storage) {
  return storageScopes.get(storage) ?? null;
}

/** @type {typeof import('./shader-source-scope.js').getScopedShaderSource} */
export function getScopedShaderSource(filename) {
  if (activeScope === null) return null;
  const entry = scopes.get(activeScope)?.get(filename);
  if (!entry) throw new Error(`Shader ${filename} is outside the verified Capsule source closure.`);
  return entry;
}

/** @type {typeof import('./shader-source-scope.js').getScopedWgslRequirements} */
export function getScopedWgslRequirements(filename) {
  // Selection may inspect variants outside the closure. Actual source loading
  // still rejects them. An old signed shader keeps its own language contract.
  return activeScope === null ? null : scopes.get(activeScope)?.get(filename)?.requiredWgslFeatures ?? null;
}

/** @type {typeof import('./shader-source-scope.js').getShaderScopeCacheKey} */
export function getShaderScopeCacheKey() {
  return `${getSharedDeviceEpoch()}:${activeScope === null ? 0 : scopeIds.get(activeScope)}`;
}

// The legacy kernels use realm-local caches. Public pipeline calls lease this
// context for their complete invocation, including iterator cleanup. Internal
// calls run on the original pipeline and never acquire a second lease.
/** @param {import("./shader-source-scope.js").ShaderSourceScope | null} scope
 * @param {AbortSignal} [signal] */
async function acquire(scope, signal) {
  if (scope !== null && !scopes.has(scope)) throw new Error('Unknown shader source scope.');
  signal?.throwIfAborted();
  const previous = pending;
  const turn = /** @type {PromiseWithResolvers<void>} */ (Promise.withResolvers());
  pending = previous.then(() => turn.promise);
  try {
    await new Promise(/** @param {(value?: void) => void} resolve */ (resolve, reject) => {
      const aborted = () => { turn.resolve(); reject(signal?.reason); };
      signal?.addEventListener('abort', aborted, { once: true });
      previous.then(() => { signal?.removeEventListener('abort', aborted); resolve(); });
    });
    signal?.throwIfAborted();
  } catch (error) { turn.resolve(); throw error; }
  activeScope = scope;
  return () => { activeScope = null; turn.resolve(); };
}

/** @type {typeof import('./shader-source-scope.js').runWithShaderSourceScope} */
export async function runWithShaderSourceScope(scope, action, signal) {
  const release = await acquire(scope, signal);
  try { return await action(); } finally { release(); }
}

/** @template T
 * @param {ShaderSourceScope | null} scope
 * @param {() => AsyncIterable<T>} action
 * @param {AbortSignal} [signal]
 * @returns {import('./shader-source-scope.js').ScopedShaderStream<T>} */
export function streamWithShaderSourceScope(scope, action, signal) {
  const waiting = new AbortController();
  /** @type {AsyncIterator<T, void, unknown> | undefined} */
  let iterator;
  /** @type {(() => void) | undefined} */
  let release;
  /** @type {unknown} */
  let reason;
  /** @type {unknown} */
  let cleanupError;
  let stopped = false, finished = false, cleanupFailed = false;
  let tail = Promise.resolve();
  const completion = /** @type {PromiseWithResolvers<void>} */ (Promise.withResolvers());
  // The owner observes closed even when a consumer abandons its iterator.
  completion.promise.catch(() => {});
  /** @template R
   * @param {() => R | PromiseLike<R>} operation
   * @returns {Promise<R>} */
  const enqueue = operation => {
    const task = tail.then(operation);
    tail = task.then(() => {}, () => {});
    return task;
  };
  async function finish() {
    if (finished) {
      if (cleanupFailed) throw cleanupError;
      return;
    }
    try {
      let result = await iterator?.return?.();
      while (iterator && result && !result.done) result = await iterator.next();
    }
    catch (error) { cleanupFailed = true; cleanupError = error; throw error; }
    finally {
      finished = true;
      signal?.removeEventListener('abort', aborted);
      release?.();
      if (cleanupFailed) completion.reject(cleanupError);
      else completion.resolve(undefined);
    }
  }
  /** @param {unknown} [error] */
  function stop(error) {
    stopped = true;
    reason ??= error;
    waiting.abort(error);
    const task = enqueue(finish);
    task.catch(() => {});
    return task;
  }
  const aborted = () => { stop(signal?.reason); };
  signal?.addEventListener('abort', aborted, { once: true });
  if (signal?.aborted) aborted();
  return {
    closed: completion.promise,
    [Symbol.asyncIterator]() { return this; },
    /** @returns {Promise<IteratorResult<T, void>>} */
    next(value) {
      return enqueue(async () => {
        try {
          if (stopped) { await finish(); if (reason) throw reason; }
          if (finished) return { done: true, value: undefined };
          if (!iterator) {
            release = await acquire(scope, waiting.signal);
            iterator = action()[Symbol.asyncIterator]();
          }
          const result = await iterator.next(value);
          if (stopped) { await finish(); if (reason) throw reason; }
          if (result.done) await finish();
          return result;
        } catch (error) {
          try { await finish(); }
          catch (cleanup) {
            if (cleanup !== error) throw new AggregateError([error, cleanup], 'Stream execution and cleanup failed.');
          }
          throw error;
        }
      });
    },
    async return(value) { await stop(); return { done: true, value: await value }; },
    async throw(error) { await stop(error); throw error; },
  };
}
