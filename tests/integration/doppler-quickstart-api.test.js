import assert from 'node:assert/strict';

import { doppler } from '../../src/index.js';
import { resolveLoadProgressHandlers } from '../../src/client/doppler-api.js';
import { getLogLevel, setLogLevel } from '../../src/debug/config.js';
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
  assert.ok(models.some((entry) => entry.modelId === 'gemma-3-270m-it-q4k-ehf16-af32'));
}

{
  const resolved = await resolveQuickstartModel('gemma3-270m');
  assert.equal(resolved.modelId, 'gemma-3-270m-it-q4k-ehf16-af32');
  assert.ok(resolved.aliases.includes('google/gemma-3-270m-it'));
  assert.match(
    buildQuickstartModelBaseUrl(resolved),
    /^https:\/\/huggingface\.co\/Clocksmith\/rdrr\/resolve\/4efe64a914892e98be50842aeb16c3b648cc68a5\/models\/gemma-3-270m-it-q4k-ehf16-af32$/
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
  const originalLogLevel = getLogLevel();
  const output = [];
  console.log = (...args) => output.push(args.join(' '));
  try {
    setLogLevel('info');
    const resolved = resolveLoadProgressHandlers({});
    assert.equal(typeof resolved.userProgress, 'function');
    assert.equal(resolved.pipelineProgress, null);
    resolved.userProgress({ phase: 'resolve', percent: 5, message: 'Resolving model' });
  } finally {
    setLogLevel(originalLogLevel);
    console.log = originalConsoleLog;
  }
  assert.ok(output.length >= 1);
  assert.ok(output.some((line) => /\[doppler\] Resolving model$/.test(line)));
}

console.log('doppler-quickstart-api.test: ok');
