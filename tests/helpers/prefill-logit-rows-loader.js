import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as nodeModule from 'node:module';

export const PREFILL_LOGIT_ROWS_KEY = Symbol.for('doppler.test.prefill-logit-rows');
export const PREFILL_LOGIT_ROWS_ANCHOR = '        const logits = await computeLogits(\n          currentHiddenBuffer,\n          numTokens,\n';

// Test-process-only teacher-forced logit rows (src/inference/pipelines/text/generator/prefill-runtime.js).
// Before the unchanged last-position logits call, the callback may evaluate the same
// computeLogits path (root mixer / final norm, last-row copy, LM head) for a prefix of
// `rows` hidden rows, which yields the GPU's logits at position rows - 1. The original
// call and its arguments are untouched; the harness checks that `compute(numTokens)`
// reproduces the production prefill logits byte for byte.
export function instrumentPrefillLogitRowsSource(text) {
  assert.equal(text.split(PREFILL_LOGIT_ROWS_ANCHOR).length - 1, 1, 'unambiguous prefill logits boundary');
  return text.replace(PREFILL_LOGIT_ROWS_ANCHOR, `        await globalThis[Symbol.for('doppler.test.prefill-logit-rows')]?.({
          numTokens, startPos, vocabSize: logitsVocabSize,
          compute: rows => computeLogits(currentHiddenBuffer, rows, getLogitsWeights(this._state),
            getLogitsConfig(this._state, 'prefill'), this._state.useGPU, this._state.debugFlags, undefined,
            debugCheckBuffer, this._state.runtimeConfig.shared.debug.probes, { lastPositionOnly: true },
            this._state.operatorDiagnostics),
        });
${PREFILL_LOGIT_ROWS_ANCHOR}`);
}

export function installPrefillLogitRowsObservation(url, expectedSha256, observe) {
  assert.equal(typeof nodeModule.registerHooks, 'function', 'requires Node synchronous module hooks (Node 22.15+)');
  assert.equal(globalThis[PREFILL_LOGIT_ROWS_KEY], undefined, 'one observer owns this test process');
  globalThis[PREFILL_LOGIT_ROWS_KEY] = observe;
  const hash = text => createHash('sha256').update(text).digest('hex');
  const receipt = { scope: 'Extra teacher-forced logit rows from the unchanged prefill logits path; no replacement', sourceUrl: url };
  const hooks = nodeModule.registerHooks({
    load(target, context, next) {
      const result = next(target, context);
      if (target !== url) return result;
      const text = typeof result.source === 'string' ? result.source : Buffer.from(result.source).toString('utf8');
      assert.equal(hash(text), expectedSha256, 'observation must identify the exact source');
      const instrumented = instrumentPrefillLogitRowsSource(text);
      receipt.originalSha256 = expectedSha256;
      receipt.instrumentedSha256 = hash(instrumented);
      return { ...result, source: instrumented };
    },
  });
  return { receipt, restore() { hooks.deregister(); delete globalThis[PREFILL_LOGIT_ROWS_KEY]; } };
}
