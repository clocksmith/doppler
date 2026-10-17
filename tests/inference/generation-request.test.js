import assert from 'node:assert/strict';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { resolveTextGenerationRequest, generationRequestEvidence } from '../../src/inference/pipelines/text/generation-request.js';
const runtime = createDopplerConfig().runtime;
const request = resolveTextGenerationRequest({ presencePenalty: 0.75 }, runtime);
assert.equal(generationRequestEvidence(request).presencePenalty, 0.75);
runtime.inference.sampling.presencePenalty = 1;
assert.equal(resolveTextGenerationRequest(request, runtime), request, 'a locally resolved request is consumed without re-reading defaults');
assert.throws(() => { request.stopSequences.push('changed'); }, TypeError);
for (const invalid of [{ seed: null }, { stopSequences: null }, { useChatTemplate: null }, { useSpeculative: null }, { maxTokens: 1.9 }]) {
  assert.throws(() => resolveTextGenerationRequest(invalid, runtime));
}
assert.throws(() => generationRequestEvidence({ ...request }), /resolved request/, 'untrusted copies must pass normalization');

assert.throws(() => resolveTextGenerationRequest({ useSpeculative: true }, runtime), /explicit supported speculation policy/);
const policy = { mode: 'lookup', tokens: 3, maxMatchTokens: 4, minMatchTokens: 3, maxRejectedBlocks: 1 };
const lookup = resolveTextGenerationRequest({ speculation: policy }, runtime);
policy.tokens = 9;
assert.equal(generationRequestEvidence(lookup).speculation.tokens, 3);
assert(Object.isFrozen(lookup.speculation));
runtime.inference.session.speculation = lookup.speculation;
assert.equal(resolveTextGenerationRequest({ speculation: null }, runtime).speculation, null);
assert.equal(resolveTextGenerationRequest({}, runtime).speculation.mode, 'lookup');
