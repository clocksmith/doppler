import { cloneGraph } from './dtype.js';
import { ATTENTION_F16KV_TO_F32_ENTRIES, deriveMappedKernelEntry } from './validation.js';

/** @type {typeof import('./kv-storage.js').widenAttentionKvToF32} */
export function widenAttentionKvToF32(graph) {
  const replacements = new Map(ATTENTION_F16KV_TO_F32_ENTRIES);
  if (!Object.values(graph.kernels).some(entry => replacements.has(entry.kernel))) return null;
  const result = cloneGraph(graph);
  for (const [key, entry] of Object.entries(result.kernels)) {
    const file = replacements.get(entry.kernel);
    if (file) result.kernels[key] = deriveMappedKernelEntry(entry, file);
  }
  return result;
}
