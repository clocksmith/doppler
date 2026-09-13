import { assertCapsuleOperationFields, normalizeCapsuleObservation } from '../../config/capsule-operation.js';

const encoder = new TextEncoder();
const size = value => encoder.encode(JSON.stringify(value)).length;
const requireValue = (value, message) => { if (!value) throw new Error(`Capsule stream: ${message}`); };

// Account for the reconstructed JSON output without copying its earlier values.
// Completion metadata is bounded separately when the final output is serialized.
export function createCapsuleDeltaBudget(request) {
  const operation = request.operation.name;
  const maxOutputBytes = request.limits.maxOutputBytes;
  const maxItems = operation === 'generate' ? request.options.maxTokens : request.input.texts?.length;
  if (operation === 'generate' || operation === 'embed') {
    requireValue(Number.isSafeInteger(maxItems) && maxItems > 0, 'explicit positive output item limit required');
  }
  let count = 0;
  let bytes = operation === 'generate' ? size({ text: '', tokenIds: [] }) : size({ embeddings: [] });
  let dimension = null;
  return {
    accept(value) {
      const delta = normalizeCapsuleObservation(value);
      if (operation === 'generate') {
        assertCapsuleOperationFields(delta, ['tokenIds', 'text'], 'generation delta');
        requireValue(Array.isArray(delta.tokenIds) && delta.tokenIds.every(id => Number.isSafeInteger(id) && id >= 0), 'invalid token IDs');
        requireValue(typeof delta.text === 'string' && delta.text.isWellFormed(), 'text must be a well-formed Unicode update');
        requireValue(delta.tokenIds.length > 0 || delta.text.length > 0, 'empty generation delta');
        requireValue(count + delta.tokenIds.length <= maxItems, 'output exceeds maxTokens');
        bytes += size(delta.text) - 2;
        for (const id of delta.tokenIds) { bytes += String(id).length + (count++ ? 1 : 0); }
      } else if (operation === 'embed') {
        assertCapsuleOperationFields(delta, ['itemIndex', 'item'], 'embedding delta');
        requireValue(delta.itemIndex === count, 'missing or reordered embedding item');
        requireValue(count < maxItems, 'output exceeds input texts count');
        const embedding = delta.item?.embedding;
        requireValue(Array.isArray(embedding) && embedding.length > 0 && embedding.every(Number.isFinite), 'invalid embedding');
        requireValue(dimension === null || dimension === embedding.length, 'inconsistent embedding dimensions');
        dimension = embedding.length;
        bytes += size(delta.item) + (count++ ? 1 : 0);
      } else throw new Error('Capsule stream: this operation does not declare partial deltas.');
      requireValue(bytes <= maxOutputBytes, 'output exceeds maxOutputBytes');
      return delta;
    },
  };
}
