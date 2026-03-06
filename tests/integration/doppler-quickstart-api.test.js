import assert from 'node:assert/strict';

import { doppler } from '../../src/index.js';
import { resolveLoadProgressHandlers } from '../../src/client/doppler-api.js';
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

{
  const calls = [];
  const callback = (event) => calls.push(event);
  const resolved = resolveLoadProgressHandlers({ onProgress: callback });
  assert.equal(resolved.userProgress, callback);
  assert.equal(resolved.pipelineProgress, callback);
}

{
  const originalConsoleLog = console.log;
  const output = [];
  console.log = (...args) => output.push(args.join(' '));
  try {
    const resolved = resolveLoadProgressHandlers({});
    assert.equal(typeof resolved.userProgress, 'function');
    assert.equal(resolved.pipelineProgress, null);
    resolved.userProgress({ phase: 'resolve', percent: 5, message: 'Resolving model' });
  } finally {
    console.log = originalConsoleLog;
  }
  assert.equal(output.length, 1);
  assert.match(output[0], /\[doppler\] Resolving model$/);
}

console.log('doppler-quickstart-api.test: ok');
