import { stableSortObject } from './stable-sort-object.js';
import { sha256Hex } from './sha256.js';

// Preserve the existing sorted-object JSON encoding, including JSON.stringify's
// integer-key ordering and coercions. It is distinct from canonicalizeJson.
/** @type {import('./sorted-json.js').hashSortedJson} */
export function hashSortedJson(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}
