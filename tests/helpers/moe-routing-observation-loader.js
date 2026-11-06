import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as nodeModule from 'node:module';

// Test-process-only readbacks after the original GPU routing has completed.
export function installMoeRoutingObservation(url, expectedSha256, observe) {
  const key = Symbol.for('doppler.test.moe-routing-observation');
  assert.equal(globalThis[key], undefined);
  globalThis[key] = observe;
  const hash = text => createHash('sha256').update(text).digest('hex');
  const receipt = { scope: 'Original GPU top-k buffers observed without replacement', sourceUrl: url };
  const hooks = nodeModule.registerHooks({
    load(target, context, next) {
      const result = next(target, context);
      if (target !== url) return result;
      let text = typeof result.source === 'string' ? result.source : Buffer.from(result.source).toString('utf8');
      assert.equal(hash(text), expectedSha256);
      const anchor = '  perfLog(`MoE L${layerIdx} topk`, stepStart, {';
      assert.equal(text.split(anchor).length - 1, 1, 'unambiguous original top-k boundary');
      text = text.replace(anchor, `  await globalThis[Symbol.for('doppler.test.moe-routing-observation')]?.({
    layer: layerIdx, numTokens, topK, activationDtype, indicesBuffer, weightsBuffer
  });\n${anchor}`);
      receipt.originalSha256 = expectedSha256;
      receipt.instrumentedSha256 = hash(text);
      return { ...result, source: text };
    },
  });
  return { receipt, restore() { hooks.deregister(); delete globalThis[key]; } };
}
