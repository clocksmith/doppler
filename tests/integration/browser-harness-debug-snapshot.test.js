import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

const { trace, log } = await import('../../src/debug/index.js');
const { runBrowserSuite } = await import('../../src/inference/browser-harness.js');

function createHarnessOverride(marker) {
  let decodeTokens = 0;
  const pipeline = {
    async *generate(_promptInput, options = {}) {
      trace.sample(`DEBUG_SNAPSHOT_TRACE:${marker}`);
      log.info('HarnessOverride', `DEBUG_SNAPSHOT_MARKER:${marker}`);
      options.onToken?.(1, 'ok');
      decodeTokens += 1;
      yield 'ok';
    },
    getStats() {
      return {
        prefillTimeMs: 1,
        ttftMs: 1,
        decodeTimeMs: 1,
        prefillTokens: 1,
        decodeTokens,
        decodeProfileSteps: [],
      };
    },
    reset() {},
    async unload() {},
  };

  return {
    modelLoadMs: 1,
    manifest: {
      modelId: 'debug-snapshot-fixture',
      modelType: 'transformer',
      architecture: {
        numLayers: 1,
        hiddenSize: 8,
        intermediateSize: 16,
        numAttentionHeads: 1,
        numKeyValueHeads: 1,
        headDim: 8,
        vocabSize: 32,
        maxSeqLen: 32,
      },
      inference: {
        attention: {
          queryPreAttnScalar: 1,
        },
      },
    },
    pipeline,
  };
}

async function readReport(result) {
  const reportPath = path.isAbsolute(result.reportInfo.path)
    ? result.reportInfo.path
    : path.resolve(process.cwd(), result.reportInfo.path);
  return JSON.parse(await readFile(reportPath, 'utf8'));
}

async function runDebugSnapshot(marker) {
  const result = await runBrowserSuite({
    suite: 'debug',
    command: 'debug',
    surface: 'node',
    runtime: {
      runtimeConfig: {
        shared: {
          debug: {
            logLevel: {
              defaultLogLevel: 'verbose',
            },
            trace: {
              enabled: true,
              categories: ['sample'],
              maxDecodeSteps: 1,
            },
          },
        },
      },
    },
    harnessOverride: createHarnessOverride(marker),
  });
  return {
    result,
    report: await readReport(result),
  };
}

const first = await runDebugSnapshot('first');
assert.ok(first.result.debugSnapshot, 'debug suite result should include debugSnapshot');
assert.ok(first.report.debugSnapshot, 'saved debug report should include debugSnapshot');
assert.ok(
  first.result.debugSnapshot.recentLogs.some((entry) => entry.message.includes('DEBUG_SNAPSHOT_MARKER:first')),
  'debugSnapshot should capture the current run trace marker'
);
assert.ok(
  first.report.debugSnapshot.recentLogs.some((entry) => entry.message.includes('DEBUG_SNAPSHOT_MARKER:first')),
  'saved report debugSnapshot should capture the current run trace marker'
);

const second = await runDebugSnapshot('second');
assert.ok(
  second.result.debugSnapshot.recentLogs.some((entry) => entry.message.includes('DEBUG_SNAPSHOT_MARKER:second')),
  'second debugSnapshot should capture the second run trace marker'
);
assert.equal(
  second.result.debugSnapshot.recentLogs.some((entry) => entry.message.includes('DEBUG_SNAPSHOT_MARKER:first')),
  false,
  'debugSnapshot should be isolated per run'
);
assert.equal(
  second.report.debugSnapshot.recentLogs.some((entry) => entry.message.includes('DEBUG_SNAPSHOT_MARKER:first')),
  false,
  'saved report debugSnapshot should be isolated per run'
);

console.log('browser-harness-debug-snapshot.test: ok');
