import { normalizeCapsuleLoadingPolicy } from '../../config/capsule-loading.js';

const deadlineChecks = new WeakMap();

export function assertCapsuleLoadActive(signal) {
  if (signal) deadlineChecks.get(signal)?.();
  if (signal?.aborted) throw signal.reason ?? new DOMException('Capsule loading cancelled.', 'AbortError');
}

export function createCapsuleLoadScope(options = {}) {
  const policy = normalizeCapsuleLoadingPolicy(options);
  const parent = options.signal;
  if (parent != null && (typeof parent.aborted !== 'boolean' || typeof parent.addEventListener !== 'function'
    || typeof parent.removeEventListener !== 'function')) throw new Error('Capsule loading signal must be an AbortSignal.');
  if (options.onLoadProgress != null && typeof options.onLoadProgress !== 'function') {
    throw new Error('Capsule onLoadProgress must be a function.');
  }
  assertCapsuleLoadActive(parent);
  const controller = new AbortController();
  const cancel = () => controller.abort(parent.reason);
  parent?.addEventListener('abort', cancel, { once: true });
  const expire = () => {
    controller.abort(new DOMException('Capsule loading deadline exceeded.', 'TimeoutError'));
  };
  const deadline = policy.loadTimeoutMs === null ? null : performance.now() + policy.loadTimeoutMs;
  deadlineChecks.set(controller.signal, () => {
    if (parent) deadlineChecks.get(parent)?.();
    if (deadline !== null && performance.now() >= deadline) expire();
  });
  const timer = policy.loadTimeoutMs === null ? null : setTimeout(expire, policy.loadTimeoutMs);
  return {
    options: { ...options, ...policy, signal: controller.signal },
    abort(reason) { controller.abort(reason); },
    close() {
      if (timer !== null) clearTimeout(timer);
      parent?.removeEventListener('abort', cancel);
      deadlineChecks.delete(controller.signal);
    },
  };
}

// Used only for byte acquisition, never to abandon a pending resource constructor.
export async function waitForCapsuleRead(task, signal) {
  if (!signal) return task;
  let cancel;
  try {
    return await Promise.race([task, new Promise((_, reject) => {
      cancel = () => reject(signal.reason ?? new DOMException('Capsule loading cancelled.', 'AbortError'));
      signal.addEventListener('abort', cancel, { once: true });
      if (signal.aborted) cancel();
    })]);
  } finally { signal.removeEventListener('abort', cancel); }
}

export async function fetchCapsuleBytes(url, { signal, onLoadProgress }, { phase, artifactId, sizeBytes, maxBytes }) {
  assertCapsuleLoadActive(signal);
  const response = await fetch(url, { signal });
  let reader;
  let completed = false;
  try {
    assertCapsuleLoadActive(signal);
    if (!response.ok) throw new Error(`Capsule ${phase} fetch failed (${response.status}) for ${url}.`);
    reader = response.body?.getReader();
    if (!reader) throw new Error(`Capsule ${phase} response has no readable body.`);
    // Artifact sizes come from signed metadata, never from an HTTP header.
    const output = sizeBytes === null ? null : new Uint8Array(sizeBytes);
    const chunks = [];
    let loadedBytes = 0;
    onLoadProgress?.({ phase, artifactId, loadedBytes, totalBytes: sizeBytes });
    while (true) {
      const { value, done } = await waitForCapsuleRead(reader.read(), signal);
      assertCapsuleLoadActive(signal);
      if (done) break;
      if (!(value instanceof Uint8Array)) throw new Error('Capsule response must contain bytes.');
      if (loadedBytes + value.byteLength > maxBytes) throw new Error(`Capsule ${phase} exceeds its byte limit.`);
      if (output) output.set(value, loadedBytes);
      else chunks.push(value.slice());
      loadedBytes += value.byteLength;
      onLoadProgress?.({ phase, artifactId, loadedBytes, totalBytes: sizeBytes });
    }
    if (sizeBytes !== null && loadedBytes !== sizeBytes) throw new Error(`Capsule artifact size mismatch for ${url}.`);
    const bytes = output ?? new Uint8Array(loadedBytes);
    if (!output) {
      let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    }
    completed = true;
    return bytes;
  } finally {
    if (!completed) {
      if (reader) await reader.cancel().catch(() => {});
      else await response.body?.cancel().catch(() => {});
    }
    reader?.releaseLock();
  }
}

export async function fetchCapsuleMetadata(url, options) {
  const policy = normalizeCapsuleLoadingPolicy(options);
  const bytes = await fetchCapsuleBytes(url, options, { phase: 'metadata', artifactId: null,
    sizeBytes: null, maxBytes: policy.maxMetadataBytes });
  return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
}
