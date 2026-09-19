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
