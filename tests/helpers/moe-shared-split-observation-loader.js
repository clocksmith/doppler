import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as nodeModule from 'node:module';

export const MOE_SHARED_SPLIT_KEY = Symbol.for('doppler.test.moe-shared-split-observation');
export const MOE_SHARED_SPLIT_ANCHOR = "  const routed = createTensor(outputBuffer, inputTensor.dtype, [...inputTensor.shape], 'moe_ffn_output');";

// Test-process-only readback of the routed-expert output before the Qwen shared
// expert is added (src/inference/pipelines/text/ffn/moe.js). It awaits a callback
// with the original tensor and never replaces a buffer or changes arithmetic, so
// a same-input audit can check routed experts and the shared expert separately.
export function instrumentMoeSharedSplitSource(text) {
  assert.equal(text.split(MOE_SHARED_SPLIT_ANCHOR).length - 1, 1, 'unambiguous routed-output boundary');
  return text.replace(MOE_SHARED_SPLIT_ANCHOR, `${MOE_SHARED_SPLIT_ANCHOR}
  await globalThis[Symbol.for('doppler.test.moe-shared-split-observation')]?.({
    layer: layerIdx, numTokens, sharedSlot: Boolean(sharedSlot), routed, input: inputTensor
  });`);
}

export function installMoeSharedSplitObservation(url, expectedSha256, observe) {
  assert.equal(typeof nodeModule.registerHooks, 'function', 'requires Node synchronous module hooks (Node 22.15+)');
  assert.equal(globalThis[MOE_SHARED_SPLIT_KEY], undefined, 'one observer owns this test process');
  globalThis[MOE_SHARED_SPLIT_KEY] = observe;
  const hash = text => createHash('sha256').update(text).digest('hex');
  const receipt = { scope: 'Routed-expert output observed before the shared-expert add; no replacement', sourceUrl: url };
  const hooks = nodeModule.registerHooks({
    load(target, context, next) {
      const result = next(target, context);
      if (target !== url) return result;
      const text = typeof result.source === 'string' ? result.source : Buffer.from(result.source).toString('utf8');
      assert.equal(hash(text), expectedSha256, 'observation must identify the exact source');
      const instrumented = instrumentMoeSharedSplitSource(text);
      receipt.originalSha256 = expectedSha256;
      receipt.instrumentedSha256 = hash(instrumented);
      return { ...result, source: instrumented };
    },
  });
  return { receipt, restore() { hooks.deregister(); delete globalThis[MOE_SHARED_SPLIT_KEY]; } };
}
