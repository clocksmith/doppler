// Exact qwen4_exp n-gram row addresses (Qwen4ExpTextNGramEmbedding.forward).
//
// Application-input preprocessing on token ids before any runtime tensor
// exists: addresses select host-resident table rows. Token-times-multiplier
// products reach about 2^63, so the hash uses BigInt; JavaScript Number cannot
// represent it. The boundary token (the source eos_token_id) separates n-gram
// segments; it is unrelated to generation stopping.
//
// For history index j and shift k, the shifted token is token[j - k] when no
// boundary token occupies positions j - k .. j - 1, and the boundary token
// otherwise (_shift_right_ignore_eos). A sequence starts with ngramSize - 1
// boundary tokens of context. For order n, mixed = XOR over p < n of
// shifted_p * multiplier_p, and head h of that order reads row
// (mixed mod headVocabSizes[h]) + headOffsets[h].

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[NgramAddresses] ${message}`);
}

export class NgramTokenHistory {
  /** @type {Uint32Array} */
  #tokens;
  #length = 0;
  #epoch = 0;

  /**
   * @param {import('../../../../config/schema/ngram-embedding.schema.js').ResolvedNgramEmbedding} contract
   * @param {import('../../../../config/schema/ngram-embedding.schema.js').ResolvedNgramEmbeddingLayer} layer
   * @param {number} [initialCapacity]
   */
  constructor(contract, layer, initialCapacity = 256) {
    if (!contract || contract.semantics !== 'qwen4_exp_ple') fail('a resolved qwen4_exp_ple contract is required.');
    if (!layer || !contract.layers.includes(layer)) fail('layer must be one of the contract layers.');
    this.contract = contract;
    this.layer = layer;
    this.context = contract.ngramSize - 1;
    this.#tokens = new Uint32Array(Math.max(initialCapacity, this.context + 1));
    this.multipliers = layer.multipliers;
    this.sizes = layer.headVocabSizes.map(BigInt);
    this.offsets = layer.headOffsets;
  }

  get length() {
    return this.#length;
  }

  get epoch() {
    return this.#epoch;
  }

  /** Token at sequence index i, or the boundary token for the leading context (i < 0). @param {number} i */
  #token(i) {
    return i < 0 ? this.contract.boundaryTokenId : this.#tokens[i];
  }

  /** @param {number} needed */
  #reserve(needed) {
    if (needed <= this.#tokens.length) return;
    const next = new Uint32Array(Math.max(needed, this.#tokens.length * 2));
    next.set(this.#tokens.subarray(0, this.#length));
    this.#tokens = next;
  }

  /**
   * Append tokens and return their addresses, [tokens.length, headCount] rows.
   * @type {import('./ngram-addresses.js').NgramTokenHistory['append']}
   */
  append(tokens) {
    const count = tokens.length;
    const { boundaryTokenId: boundary, unigramVocabSize, headCount, headsPerNgram, ngramSize } = this.contract;
    for (let i = 0; i < count; i++) {
      const token = tokens[i];
      if (!Number.isSafeInteger(token) || token < 0 || token >= unigramVocabSize) {
        fail(`token ${String(token)} at offset ${i} is outside the ${unigramVocabSize}-token vocabulary.`);
      }
    }
    this.#reserve(this.#length + count);
    const start = this.#length;
    for (let i = 0; i < count; i++) this.#tokens[start + i] = tokens[i];
    const addresses = new Uint32Array(count * headCount);
    const shifted = new Array(ngramSize);
    for (let i = 0; i < count; i++) {
      const j = start + i;
      shifted[0] = BigInt(this.#token(j));
      for (let k = 1; k < ngramSize; k++) {
        let crossed = false;
        for (let back = 1; back <= k; back++) {
          if (this.#token(j - back) === boundary) {
            crossed = true;
            break;
          }
        }
        shifted[k] = BigInt(crossed ? boundary : this.#token(j - k));
      }
      for (let n = 2; n <= ngramSize; n++) {
        let mixed = shifted[0] * this.multipliers[0];
        for (let p = 1; p < n; p++) mixed ^= shifted[p] * this.multipliers[p];
        const firstHead = (n - 2) * headsPerNgram;
        for (let h = firstHead; h < firstHead + headsPerNgram; h++) {
          addresses[i * headCount + h] = Number(mixed % this.sizes[h]) + this.offsets[h];
        }
      }
    }
    this.#length = start + count;
    return addresses;
  }

  /** @type {import('./ngram-addresses.js').NgramTokenHistory['truncate']} */
  truncate(length) {
    if (!Number.isSafeInteger(length) || length < 0) fail('truncate length must be a non-negative integer.');
    if (length < this.#length) this.#length = length;
    this.#epoch += 1;
  }

  /** @type {import('./ngram-addresses.js').NgramTokenHistory['reset']} */
  reset() {
    this.truncate(0);
  }

  /** @type {import('./ngram-addresses.js').NgramTokenHistory['checkpoint']} */
  checkpoint() {
    return Object.freeze({ epoch: this.#epoch, length: this.#length });
  }

  /** @type {import('./ngram-addresses.js').NgramTokenHistory['restore']} */
  restore(checkpoint) {
    if (checkpoint?.epoch !== this.#epoch || !Number.isSafeInteger(checkpoint.length) || checkpoint.length > this.#length) {
      fail('checkpoint is not restorable: it predates a truncate, reset or restore, or exceeds the history.');
    }
    this.#length = checkpoint.length;
    this.#epoch += 1;
  }
}
