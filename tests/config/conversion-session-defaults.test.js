import assert from 'node:assert/strict';
import fs from 'node:fs';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

const EXPLICIT_TEXT_DEFAULTS = [
  {
    path: 'src/config/conversion/gemma3/gemma-3-1b-it-f16-af32.json',
    computeDefaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 4,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      readbackMode: 'sequential',
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
    },
  },
  {
    path: 'src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json',
    computeDefaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 1,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      readbackMode: 'sequential',
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
    },
  },
  {
    path: 'src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json',
    computeDefaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 4,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      readbackMode: 'sequential',
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
    },
  },
  {
    path: 'src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json',
    computeDefaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 4,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      readbackMode: 'sequential',
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
    },
  },
  {
    path: 'src/config/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json',
    computeDefaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 4,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      readbackMode: 'sequential',
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
    },
  },
  {
    path: 'src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json',
    computeDefaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 8,
      stopCheckMode: 'batch',
      readbackInterval: 8,
      readbackMode: 'overlapped',
      ringTokens: 2,
      ringStop: 1,
      ringStaging: 2,
      disableCommandBatching: false,
    },
  },
  {
    path: 'src/config/conversion/gemma4/gemma-4-moe-q4k-ehf16-af32.json',
    computeDefaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 4,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      readbackMode: 'sequential',
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
    },
  },
  {
    path: 'src/config/conversion/gpt-oss-20b-f16-xmxfp4.json',
    computeDefaults: {
      activationDtype: 'f16',
      mathDtype: 'f16',
      accumDtype: 'f16',
      outputDtype: 'f16',
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 4,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      readbackMode: 'sequential',
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
    },
  },
  {
    path: 'src/config/conversion/janus/janus-pro-1b-text-q4k-ehaf16.json',
    computeDefaults: {
      activationDtype: 'f16',
      mathDtype: 'f16',
      accumDtype: 'f16',
      outputDtype: 'f16',
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 4,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      readbackMode: 'sequential',
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
    },
  },
  {
    path: 'src/config/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json',
    computeDefaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 8,
      stopCheckMode: 'batch',
      readbackInterval: 8,
      readbackMode: 'sequential',
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
    },
  },
  {
    path: 'src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json',
    computeDefaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      maxSeqLen: 8192,
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 4,
      stopCheckMode: 'batch',
      readbackInterval: 32,
      readbackMode: 'sequential',
      submitLatencyThresholdMs: null,
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
    },
    speculation: {
      mode: 'self',
      tokens: 1,
      verify: 'greedy',
      threshold: null,
      rollbackOnReject: true,
    },
  },
  {
    path: 'src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json',
    computeDefaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
    kvcache: {
      kvDtype: 'f32',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 4,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      readbackMode: 'sequential',
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
    },
  },
  {
    path: 'src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json',
    computeDefaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      maxSeqLen: 8192,
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 8,
      stopCheckMode: 'batch',
      readbackInterval: 32,
      readbackMode: 'sequential',
      submitLatencyThresholdMs: null,
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
    },
    speculation: {
      mode: 'self',
      tokens: 1,
      verify: 'greedy',
      threshold: null,
      rollbackOnReject: true,
    },
  },
];

const EXPLICIT_NULL_OR_DISABLED = [
  {
    path: 'src/config/conversion/sana/sana-sprint-0.6b-f16.json',
    session: {
      computeDefaults: {
        activationDtype: 'f16',
        mathDtype: 'f16',
        accumDtype: 'f16',
        outputDtype: 'f16',
      },
      kvcache: null,
      decodeLoop: null,
    },
  },
];

for (const fixture of EXPLICIT_TEXT_DEFAULTS) {
  const config = readJson(fixture.path);
  assert.ok(config.session && typeof config.session === 'object', `${fixture.path} session`);
  assert.deepEqual(config.session.compute?.defaults, fixture.computeDefaults, `${fixture.path} compute.defaults`);
  assert.deepEqual(config.session.kvcache, fixture.kvcache, `${fixture.path} kvcache`);
  assert.deepEqual(config.session.decodeLoop, fixture.decodeLoop, `${fixture.path} decodeLoop`);
  if (fixture.speculation) {
    assert.deepEqual(config.session.speculation, fixture.speculation, `${fixture.path} speculation`);
  }
}

for (const fixture of EXPLICIT_NULL_OR_DISABLED) {
  const config = readJson(fixture.path);
  if (fixture.session === null) {
    assert.equal(config.session, null, `${fixture.path} session`);
    continue;
  }
  assert.ok(config.session && typeof config.session === 'object', `${fixture.path} session`);
  assert.deepEqual(config.session.compute?.defaults, fixture.session.computeDefaults, `${fixture.path} compute.defaults`);
  assert.deepEqual(config.session.kvcache, fixture.session.kvcache, `${fixture.path} kvcache`);
  assert.equal(config.session.decodeLoop, fixture.session.decodeLoop, `${fixture.path} decodeLoop`);
}

console.log('conversion-session.test: ok');
