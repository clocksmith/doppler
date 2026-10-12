import { createSha256Hasher } from '../formats/sha256.js';
import { GENERATION_CONTRACT } from '../config/generation-contract.js';

/** @type {import('./canonical-contract-codec.js').contractAssert} */
export function contractAssert(ok, message) {
  if (!ok) throw Object.assign(new Error(message), { code: 'DOPPLER_CONTRACT_INVALID' });
}

/** @type {import('./canonical-contract-codec.js').count} */
export function count(value, label, maximum = Number.MAX_SAFE_INTEGER) {
  contractAssert(typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 && value <= maximum, `Invalid ${label}.`);
  return value;
}

/** @type {import('./canonical-contract-codec.js').hash} */
export function hash(value, label) {
  contractAssert(typeof value === 'string' && /^sha256:[a-f0-9]{64}$/.test(value), `Invalid ${label} digest.`);
  return value;
}

/** @type {import('./canonical-contract-codec.js').scalarString} */
export function scalarString(value) {
  contractAssert(typeof value === 'string', 'Expected Unicode string.');
  for (let i = 0; i < value.length; i++) {
    const unit = value.charCodeAt(i);
    if (unit >= 0xd800 && unit <= 0xdbff) {
      const next = value.charCodeAt(++i);
      contractAssert(next >= 0xdc00 && next <= 0xdfff, 'Unpaired Unicode surrogate.');
    } else contractAssert(unit < 0xdc00 || unit > 0xdfff, 'Unpaired Unicode surrogate.');
  }
  return value;
}

/** @type {import('./canonical-contract-codec.js').exactRecord} */
export function exactRecord(value, keys) {
  contractAssert(value !== null && typeof value === 'object'
    && [Object.prototype, null].includes(Object.getPrototypeOf(value)), 'Expected plain contract record.');
  const own = Reflect.ownKeys(value);
  contractAssert(own.length === keys.length && own.every(key => typeof key === 'string' && keys.includes(key)),
    'Missing or unknown contract field.');
  for (const key of own) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    contractAssert(descriptor && descriptor.enumerable && Object.hasOwn(descriptor, 'value'), 'Contract accessors are forbidden.');
  }
  return value;
}

// RFC8785 uses ECMAScript number serialization and UTF-16 key ordering. Validate
// Unicode, descriptors and cumulative bounds before building the canonical text.
/** @type {import('./canonical-contract-codec.js').canonicalMetadata} */
export function canonicalMetadata(value) {
  const limits = GENERATION_CONTRACT.canonical.metadataLimits;
  let nodes = 0, units = 0;
  /** @param {unknown} item @param {number} depth */
  const visit = (item, depth) => {
    contractAssert(++nodes <= limits.maxNodes && depth <= limits.maxDepth, 'Metadata structure exceeds admission.');
    if (item === null) { units += 4; return; }
    if (typeof item === 'string') {
      contractAssert(item.length <= limits.maxBytes, 'Metadata string exceeds admission.');
      units += JSON.stringify(scalarString(item)).length;
    } else if (typeof item === 'number') {
      contractAssert(Number.isFinite(item), 'Metadata number must be finite.');
      units += JSON.stringify(item).length;
    } else if (typeof item === 'boolean') units += item ? 4 : 5;
    else if (Array.isArray(item)) {
      contractAssert(item.length <= limits.maxNodes, 'Metadata array exceeds admission.');
      contractAssert(Reflect.ownKeys(item).length === item.length + 1, 'Sparse or extended metadata array.');
      units += 2 + Math.max(0, item.length - 1);
      for (let i = 0; i < item.length; i++) {
        const property = Object.getOwnPropertyDescriptor(item, String(i));
        contractAssert(property && Object.hasOwn(property, 'value'), 'Sparse or accessor metadata array.');
        visit(property.value, depth + 1);
      }
    } else {
      contractAssert(item && typeof item === 'object', 'Metadata must be JSON.');
      const keys = Reflect.ownKeys(item);
      contractAssert(keys.length <= limits.maxNodes, 'Metadata record exceeds admission.');
      exactRecord(item, keys);
      const record = /** @type {Record<string, unknown>} */ (item);
      units += 2 + Math.max(0, keys.length - 1);
      for (const key of keys) { visit(key, depth + 1); units++; visit(record[String(key)], depth + 1); }
    }
    contractAssert(units <= limits.maxBytes, 'Metadata exceeds admission.');
  };
  visit(value, 0);
  /** @param {unknown} item @returns {string} */
  const serialize = item => Array.isArray(item) ? `[${item.map(serialize).join(',')}]`
    : item !== null && typeof item === 'object'
      ? `{${Object.keys(item).sort().map(key => `${JSON.stringify(key)}:${serialize(/** @type {Record<string, unknown>} */ (item)[key])}`).join(',')}}`
      : JSON.stringify(item);
  const text = serialize(value);
  contractAssert(new TextEncoder().encode(text).byteLength <= limits.maxBytes, 'Metadata UTF-8 exceeds admission.');
  return text;
}

/** @type {import('./canonical-contract-codec.js').metadataDigest} */
export function metadataDigest(schema, body) {
  scalarString(schema);
  contractAssert(schema.length > 0 && schema.length <= 256 && !schema.includes('\0'), 'Invalid digest schema.');
  const digest = createSha256Hasher();
  digest.update(new TextEncoder().encode(schema + '\0'));
  digest.update(new TextEncoder().encode(canonicalMetadata(body)));
  return `sha256:${digest.digestHex()}`;
}

/** @type {import('./canonical-contract-codec.js').admitTokens} */
export function admitTokens(tokens, vocabSize, maxTokens) {
  count(vocabSize, 'vocabulary size', 0x100000000);
  contractAssert(vocabSize > 0, 'Empty vocabulary.');
  count(maxTokens, 'token array bound');
  contractAssert(Array.isArray(tokens) && tokens.length <= maxTokens, 'Token array exceeds admission.');
  contractAssert(Reflect.ownKeys(tokens).length === tokens.length + 1, 'Sparse or extended token array.');
  for (let i = 0; i < tokens.length; i++) {
    const property = Object.getOwnPropertyDescriptor(tokens, String(i));
    contractAssert(property && Object.hasOwn(property, 'value'), 'Sparse or accessor token array.');
    count(property.value, 'token ID', vocabSize - 1);
  }
  return Object.freeze(tokens.slice());
}

/** @type {import('./canonical-contract-codec.js').tokenDigest} */
export function tokenDigest(tokens, vocabSize, maxTokens) {
  const admitted = admitTokens(tokens, vocabSize, maxTokens);
  const digest = createSha256Hasher();
  digest.update(new TextEncoder().encode('doppler.tokens/v1\0'));
  const header = new DataView(new ArrayBuffer(8));
  header.setBigUint64(0, BigInt(admitted.length), true);
  digest.update(new Uint8Array(header.buffer));
  const word = new DataView(new ArrayBuffer(4));
  for (const id of admitted) { word.setUint32(0, id, true); digest.update(new Uint8Array(word.buffer)); }
  return `sha256:${digest.digestHex()}`;
}

/** @type {import('./canonical-contract-codec.js').immutableRecord} */
export function immutableRecord(value) {
  canonicalMetadata(value);
  const copy = structuredClone(value);
  /** @param {unknown} item */
  const freeze = item => {
    if (item && typeof item === 'object') { Object.values(item).forEach(freeze); Object.freeze(item); }
  };
  freeze(copy);
  return copy;
}
