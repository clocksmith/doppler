import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as nodeModule from 'node:module';

// Test-process-only source instrumentation. It adds awaited readbacks through
// a callback, never replaces a tensor or changes arithmetic, and identifies
// both the original and loaded source. Install before importing the pipeline.
export function installHyperConnectionObservation(url, expectedSha256, observe) {
  assert.equal(typeof nodeModule.registerHooks, 'function', 'This provisioned trace requires Node synchronous module hooks (Node 22.15+).');
  const key = Symbol.for('doppler.test.hyper-connection-observation');
  assert.equal(globalThis[key], undefined, 'one observer owns this test process');
  globalThis[key] = observe;
  const hash = text => createHash('sha256').update(text).digest('hex');
  const receipt = { scope: 'Test-only awaited observations; no arithmetic replacement', sourceUrl: url };
  const hooks = nodeModule.registerHooks({
    load(target, context, next) {
      const result = next(target, context);
      if (target !== url) return result;
      let text = typeof result.source === 'string' ? result.source : Buffer.from(result.source).toString('utf8');
      assert.equal(hash(text), expectedSha256, 'trace must identify the exact source');
      const replacements = [
        ['const ops = operators(target);\n  const { streamCount', "await globalThis[Symbol.for('doppler.test.hyper-connection-observation')]?.({ role, stage: 'input', tensor: streams, numTokens });\n  const ops = operators(target);\n  const { streamCount"],
        ["const normed = keep(await ops.norm(streams, weights.norm, { ...geometry, eps: normEps }));", 'norm'],
        ['const down = keep(await project(`${role}_down`, normed, weights.down, projectionRank, width));', 'down'],
        ["const lowRank = keep(await ops.activate(down, { mode: 'low_rank_silu', rows: numTokens, columns: projectionRank, streamCount }));", 'lowRank'],
        ['const up = keep(await project(`${role}_up`, lowRank, weights.up, width, projectionRank));', 'up'],
        ["const mix = keep(await ops.activate(up, { mode: 'mix_sigmoid', rows: numTokens, columns: width, streamCount }));", 'mix'],
        ['const branchInput = await ops.branch(mix, normed, geometry);', 'branchInput'],
        ['const logits = keep(await project(`${role}_inject`, normed, weights.inject, streamCount, width));', 'logits'],
        ["const gate = await ops.activate(logits, { mode: 'inject_gate', rows: numTokens, columns: streamCount, streamCount });", 'gate'],
      ];
      for (let i = 0; i < replacements.length; i++) {
        const [anchor, value] = replacements[i];
        assert.equal(text.split(anchor).length - 1, 1, `unambiguous trace anchor ${i}`);
        const tensor = value === 'norm' ? 'normed' : value;
        text = text.replace(anchor, i === 0 ? value : `${anchor}\n    await globalThis[Symbol.for('doppler.test.hyper-connection-observation')]?.({ role, stage: '${value}', tensor: ${tensor}, numTokens });`);
      }
      for (const [anchor, stage, tensor] of [
        ['output = await branch(branchInput);', 'branchOutput', 'output'],
        ['const injected = await operators(target).inject(streams, output, gate,\n      { numTokens, streamCount: hyper.streamCount, streamWidth: hyper.streamWidth });', 'injected', 'injected'],
      ]) {
        assert.equal(text.split(anchor).length - 1, 1, `unambiguous ${stage} trace anchor`);
        text = text.replace(anchor, `${anchor}\n    await globalThis[Symbol.for('doppler.test.hyper-connection-observation')]?.({ role, stage: '${stage}', tensor: ${tensor}, numTokens });`);
      }
      receipt.originalSha256 = expectedSha256;
      receipt.instrumentedSha256 = hash(text);
      return { ...result, source: text };
    },
  });
  return { receipt, restore() { hooks.deregister(); delete globalThis[key]; } };
}
