import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// The actual observation adapter runs with a stubbed probe transport. This is
// ownership/geometry coverage, not GPU execution or evidence of a state fix.
const source = await readFile(new URL('../../src/inference/pipelines/text/linear-attention.js', import.meta.url), 'utf8');
const start = source.indexOf('async function observeLinearAttentionState(');
const end = source.indexOf('async function syncLayerRuntimeStateFromGPU(', start);
assert(start >= 0 && end > start);
const createObserver = new Function('runProbes', `${source.slice(start, end)}\nreturn observeLinearAttentionState;`);
const fields = ['convWeight', 'dtBias', 'aLog', 'normWeight', 'convState', 'recurrentState'];
const names = ['conv_weight', 'dt_bias', 'a_log', 'norm_weight', 'conv_input', 'recurrent_input'];
const state = Object.fromEntries(fields.flatMap((field, index) => [
  [field, new Float32Array(index + 1).fill(index + 1)],
  [`${field}GPU`, { id: field, size: 256 }],
]));
for (const recorder of [null, { id: 'recorder' }]) {
  for (const failAt of [-1, 0, 5]) {
    const calls = [];
    const failure = new Error('state probe failed');
    const options = { layerIdx: 3, recorder, debugProbes: [{ stage: 'linear_state_conv_input' }],
      operatorDiagnostics: { enabled: true } };
    const observe = createObserver(async (stage, buffer, config) => {
      const index = calls.length;
      calls.push(stage);
      assert.equal(stage, `linear_state_${names[index]}`);
      assert.equal(buffer, state[`${fields[index]}GPU`]);
      assert.equal(config.hiddenSize, index + 1, 'Use logical elements, not pooled buffer capacity');
      assert.equal(config.numTokens, 1);
      assert.equal(config.dtype, 'f32');
      assert.equal(config.layerIdx, options.layerIdx);
      assert.equal(config.probes, options.debugProbes);
      assert.equal(config.recorder, recorder);
      assert.equal(config.operatorDiagnostics, options.operatorDiagnostics);
      if (index === failAt) throw failure;
    });
    if (failAt >= 0) await assert.rejects(observe(state, options), error => error === failure);
    else await observe(state, options);
    assert.equal(calls.length, failAt >= 0 ? failAt + 1 : fields.length);
    for (const [index, field] of fields.entries()) {
      assert.deepEqual([...state[field]], Array(index + 1).fill(index + 1));
      assert.equal(state[`${field}GPU`].id, field);
    }
  }
}
assert.match(source, /try \{\s*if \(options\.debugProbes\?\.some\([\s\S]*?await observeLinearAttentionState\(layerState, options\);/);
const guardStart = source.indexOf('    if (options.debugProbes?.some(');
const guardEnd = source.indexOf("    await runProbes('linear_qkv_proj'", guardStart);
assert(guardStart > 0 && guardEnd > guardStart);
const dispatchObservation = new Function('options', 'observeLinearAttentionState', 'layerState',
  `return (async () => { ${source.slice(guardStart, guardEnd)} })();`);
let observations = 0;
const observer = async actual => { assert.equal(actual, state); observations += 1; };
for (const options of [{}, { debugProbes: [] }, { debugProbes: [{ stage: 'linear_qkv_proj' }] }]) {
  await dispatchObservation(options, observer, state);
}
assert.equal(observations, 0, 'Ordinary execution and unrelated probes must not enter state observation');
await dispatchObservation({ debugProbes: [{ stage: 'linear_state_conv_input' }] }, observer, state);
assert.equal(observations, 1);
console.log('linear-attention-state-probes.test: ok');
