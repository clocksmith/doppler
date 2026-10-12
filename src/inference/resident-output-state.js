import { validateCanonicalGenerationPolicy } from '../config/canonical-generation-policy.js';
import { contractAssert as assert, count, metadataDigest } from '../config/canonical-contract-codec.js';

/** @type {import('./resident-output-state.js').createResidentOutputState} */
export function createResidentOutputState(raw, visible, policy, vocabSize) {
  return createOutput(raw, visible, validateCanonicalGenerationPolicy(policy, vocabSize), vocabSize, null);
}

/** @param {import('./tokenizers/bundled/incremental-decoder.js').IncrementalTokenDecoder} raw
 * @param {import('./tokenizers/bundled/incremental-decoder.js').IncrementalTokenDecoder} visible
 * @param {import('../config/generation-contract.js').CanonicalGenerationPolicy} policy
 * @param {number} vocabSize
 * @param {import('./resident-output-state.js').OutputControl | null} saved
 * @returns {import('./resident-output-state.js').ResidentOutputState} */
function createOutput(raw, visible, policy, vocabSize, saved) {
  let withheld = saved?.withheld ?? '', rawUtf16 = saved?.rawUtf16 ?? 0;
  let visibleUtf16 = saved?.visibleUtf16 ?? 0, tokens = saved?.tokens ?? 0;
  let stopped = saved?.stopped ?? false, closed = saved?.closed ?? false;
  const controls = () => ({ withheld, rawUtf16, visibleUtf16, tokens, stopped, closed });
  const bounds = () => {
    assert(rawUtf16 + (closed ? 0 : raw.pendingText().length) <= policy.maxRawOutputUtf16,
      'Raw UTF-16 output exceeds admission.');
    assert(visibleUtf16 <= policy.maxVisibleOutputUtf16, 'Visible UTF-16 output exceeds admission.');
  };
  /** @param {string} chunk */
  const consume = chunk => {
    const text = withheld + chunk;
    withheld = '';
    // Scan by completion position, then start, then configured order. This also
    // sees matches wholly inside one token, not just matches at chunk end.
    let bestEnd = Infinity, bestStart = Infinity, bestOrder = Infinity;
    for (let order = 0; order < policy.stopSequences.length; order++) {
      const stop = policy.stopSequences[order], start = text.indexOf(stop);
      if (start < 0) continue;
      const end = start + stop.length;
      if (end < bestEnd || end === bestEnd && (start < bestStart || start === bestStart && order < bestOrder)) {
        bestEnd = end; bestStart = start; bestOrder = order;
      }
    }
    let delta;
    if (bestEnd !== Infinity) { stopped = true; delta = text.slice(0, bestStart); }
    else {
      let retain = 0;
      for (const stop of policy.stopSequences) {
        for (let size = Math.min(stop.length - 1, text.length); size > retain; size--) {
          if (text.endsWith(stop.slice(0, size))) { retain = size; break; }
        }
      }
      withheld = text.slice(text.length - retain);
      delta = text.slice(0, text.length - retain);
    }
    visibleUtf16 += delta.length;
    return delta;
  };
  return {
    accept(tokenId) {
      assert(!closed && !stopped, 'Decoder successor is terminal.');
      count(tokenId, 'accepted token', vocabSize - 1);
      assert(tokens < policy.maxGeneratedTokens, 'Decoder generated-token allowance exhausted.');
      rawUtf16 += raw.push(tokenId).length;
      tokens++;
      const delta = consume(visible.push(tokenId));
      bounds();
      return { delta, textStop: stopped };
    },
    finalize(reason) {
      assert(!closed, 'Decoder already finalized.');
      rawUtf16 += raw.finish({ stopReason: reason }).length;
      const tail = visible.finish({ stopReason: reason });
      let delta = '';
      if (!stopped) {
        delta = consume(tail);
        if (!stopped) { delta += withheld; visibleUtf16 += withheld.length; withheld = ''; }
      }
      closed = true;
      bounds();
      return delta;
    },
    getUsage: controls,
    digest() {
      assert(raw.snapshot && visible.snapshot, 'Decoder snapshot is unsupported.');
      return metadataDigest('doppler.output-state/v1', { ...controls(), raw: raw.snapshot(), visible: visible.snapshot() });
    },
    clone() {
      assert(raw.clone && visible.clone && raw.snapshot && visible.snapshot, 'Decoder clone is unsupported.');
      return createOutput(raw.clone(), visible.clone(), policy, vocabSize, controls());
    },
  };
}
