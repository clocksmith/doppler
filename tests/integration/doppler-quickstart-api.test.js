import assert from 'node:assert/strict';

import { doppler } from '../../src/index.js';
import {
  buildQuickstartModelBaseUrl,
  listQuickstartModels,
  resolveQuickstartModel,
} from '../../src/client/doppler-registry.js';

assert.equal(typeof doppler, 'function');
assert.equal(typeof doppler.load, 'function');
assert.equal(typeof doppler.text, 'function');
assert.equal(typeof doppler.chat, 'function');
assert.equal(typeof doppler.chatText, 'function');
assert.equal(typeof doppler.evict, 'function');

{
  const models = await listQuickstartModels();
  assert.ok(models.some((entry) => entry.modelId === 'gemma-3-270m-it-wq4k-ef16-hf16'));
}

{
  const resolved = await resolveQuickstartModel('gemma3-270m');
  assert.equal(resolved.modelId, 'gemma-3-270m-it-wq4k-ef16-hf16');
  assert.ok(resolved.aliases.includes('google/gemma-3-270m-it'));
  assert.match(
    buildQuickstartModelBaseUrl(resolved),
    /^https:\/\/huggingface\.co\/Clocksmith\/rdrr\/resolve\/4efe64a914892e98be50842aeb16c3b648cc68a5\/models\/gemma-3-270m-it-wq4k-ef16$/
  );
}

{
  await assert.rejects(
    () => doppler.load('gemma-3-1b'),
    /Unknown quickstart model "gemma-3-1b"/
  );
}

console.log('doppler-quickstart-api.test: ok');
