// A conservative retention budget for JSON recordings. Array slots, keys, strings
// and nested records all count. It is an estimate, not a browser heap measurement.
export const retainedLimit = 256 * 1024 * 1024;
export const retainedBytes = sizes => [...sizes.values()].reduce((sum, bytes) => sum + bytes, 0);

export function estimateRecordingBytes(value, available = retainedLimit) {
  let total = 0;
  function count(item, depth) {
    if (depth > 128) throw new Error('Recording nesting exceeds the workspace limit.');
    total += typeof item === 'string' ? 16 + 2 * item.length : 16;
    if (total > available) throw new Error('The 256 MiB recording budget is full. Remove recordings or use a smaller capture.');
    if (!item || typeof item !== 'object') return;
    if (Array.isArray(item)) {
      for (const entry of item) count(entry, depth + 1);
    } else {
      for (const name of Object.keys(item)) {
        total += 2 * name.length + 32;
        count(item[name], depth + 1);
      }
    }
  }
  count(value, 0);
  return total;
}

export function retain(sizes, id, value, replaces = null) {
  const available = retainedLimit - retainedBytes(sizes) + (sizes.get(replaces) ?? 0);
  if (sizes.size - Number(sizes.has(replaces)) >= 64) throw new Error('64 recordings are open. Remove a recording to make room.');
  const total = estimateRecordingBytes(value, available);
  if (replaces) sizes.delete(replaces);
  sizes.set(id, total);
}

// Allowlist only numerical aggregates. IDs, token IDs, activations, candidates,
// arbitrary imported fields and free text can reveal the original question.
export function redactedSummary(run) {
  const metric = value => Number.isFinite(value) ? value : null;
  return { schema: 'doppler.observatory.redacted-summary/v1', answerTokens: run.tokens.length,
    wallMs: metric(run.timing.wallMs), firstTokenMs: metric(run.timing.timeToFirstTokenMs), decodeMs: metric(run.timing.decodeMs) };
}

export function downloadFile(name, text, type = 'application/json') {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([text], { type }));
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
