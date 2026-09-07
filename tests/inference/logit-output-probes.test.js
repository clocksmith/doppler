import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// Exercise the actual orchestration with stubbed GPU dispatch and observation.
// This does not execute shaders or establish physical readback correctness.
const source = await readFile(process.argv[2]
  ?? new URL('../../src/inference/pipelines/text/logits/output-transform.js', import.meta.url), 'utf8');
const imports = source.match(/import[\s\S]*?from '[^']+';/g);
assert.equal(imports.length, 4);
const body = source.replace(/import[\s\S]*?from '[^']+';/g, '')
  .replace('export async function finalizeLogitOutputTensor', 'async function finalizeLogitOutputTensor');
const instantiate = new Function('releaseBuffer', 'runProbes', 'resolveLogitOutputScale',
  'recordFinalizeLogitsTensor', 'runFinalizeLogitsTensor',
  `${body}\nreturn finalizeLogitOutputTensor;`);

for (const recorded of [false, true]) {
  for (const failProbe of [false, true]) {
    const released = [], tracked = [], observed = [];
    const tensor = { buffer: { id: 'input' }, dtype: 'f32' };
    const output = { buffer: { id: 'output' }, dtype: 'f32' };
    const probes = [{ stage: 'logits', dims: [0], tokens: [0] }];
    const recorder = recorded ? { trackTemporaryBuffer: buffer => tracked.push(buffer) } : null;
    const diagnostics = { enabled: true };
    const failure = new Error('probe readback failed');
    const run = instantiate(buffer => released.push(buffer), async (stage, buffer, options) => {
      observed.push({ stage, buffer, options });
      assert.equal(options.probes, probes, 'Configured probes must reach GPU output observation');
      if (failProbe) throw failure;
    }, config => config.logitOutputScale,
    async (actualRecorder, input) => {
      assert.equal(actualRecorder, recorder);
      assert.equal(input, tensor);
      return output;
    }, async input => {
      assert.equal(input, tensor);
      return output;
    });
    const result = run(tensor, { debugProbes: probes, logitOutputScale: 1 }, {
      recorder, numTokens: 1, vocabSize: 2, targetVocabSize: 3, operatorDiagnostics: diagnostics,
    });
    if (failProbe) await assert.rejects(result, error => error === failure);
    else assert.equal(await result, output);
    assert.equal(observed.length, 1);
    assert.equal(observed[0].stage, 'logits');
    assert.equal(observed[0].buffer, output.buffer);
    assert.equal(observed[0].options.hiddenSize, 3);
    assert.equal(observed[0].options.recorder, recorder);
    assert.equal(observed[0].options.operatorDiagnostics, diagnostics);
    const owned = failProbe ? [tensor.buffer, output.buffer] : [tensor.buffer];
    assert.deepEqual(released, recorded ? [] : owned);
    assert.deepEqual(tracked, recorded ? owned : []);
  }
}

console.log('logit-output-probes.test: ok');
