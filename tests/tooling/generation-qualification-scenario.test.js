import assert from 'node:assert/strict';
import { compareGenerationTokens } from '../../tools/generation-qualification-scenario.js';
const source = { queryIndex: 4, promptTokenIds: [10, 11], generatedTokenIds: [12, 13, 2] };
const compare = (tokenIds, prompt = source.promptTokenIds) => compareGenerationTokens(source, { tokenIds }, prompt);
assert.equal(compare([12, 13, 2]).passed, true);
assert.equal(compare([12, 13]).firstTokenMismatch, 2, 'Truncation cannot pass an exact oracle.');
assert.equal(compare([12, 13, 2, 9]).firstTokenMismatch, 3, 'Extra output cannot pass an exact oracle.');
assert.equal(compare([12, 14, 2]).firstTokenMismatch, 1);
assert.equal(compare([12, 13, 2], [10, 15]).passed, false, 'Same continuation with changed prompt tokens is not source parity.');
assert.throws(() => compare([12, NaN, 2]), /integer token arrays/);
assert.throws(() => compare([12, 1.5, 2]), /integer token arrays/);
console.log('generation-qualification-scenario.test: ok');

// Exercise the complete lifecycle with an API that rejects cancellation, as
// physical generation does. Only an expected AbortError may be accepted.
const { runGenerationQualificationScenario } = await import('../../tools/generation-qualification-scenario.js');
const moduleUrl = code => 'data:text/javascript,' + encodeURIComponent(code);
for (const failure of ['AbortError', 'TypeError']) {
  const fixtureKey = '__generationQualification' + failure;
  let device;
  const fixture = globalThis[fixtureKey] = {
    async load() {
      device = { destroyed: false, destroy() { this.destroyed = true; }, lost: Promise.resolve({ reason: 'destroyed' }) };
      const owned = device;
      return { manifest: { artifactIdentity: { sourceRepo: 'fixture', sourceRevision: 'one' } },
        advanced: { getResolvedRuntimeSession: () => ({}), tokenizePrompt: () => [10, 11], getStats: () => ({ stopReason: 'aborted' }) },
        resetGenerationState() {}, unload: async () => {},
        async generateWithEvidence() { if (owned.destroyed) throw Error('device lost'); return { tokenIds: [12, 13, 2] }; },
        async *generate(_prompt, { signal }) {
          yield 'first'; assert(signal.aborted);
          throw Object.assign(Error('cancelled fixture'), { name: failure });
        } };
    }, getDevice: () => device, destroyDevice() { device?.destroy(); }, observeInitialExecutionIdentity: () => ({})
  };
  const url = moduleUrl(`export const { load, getDevice, destroyDevice, observeInitialExecutionIdentity } = globalThis.${fixtureKey};`);
  try {
    const result = await runGenerationQualificationScenario({ apiModule: url, deviceModule: url, identityModule: url,
      modelUrl: 'fixture:', generation: {}, repeatRuns: 1, cancellation: { afterChunks: 1, maxElapsedMs: 1000 } },
    { source: { model: 'fixture', revision: 'one' }, outputs: [{ ...source, prompt: 'fixture' }] }, async () => {});
    assert.equal(result.passed, failure === 'AbortError');
    if (failure === 'AbortError') {
      assert.equal(result.cancellation.abortRejection.name, failure);
      assert.equal(result.afterDeviceLoss.comparison.passed, true);
    } else assert.equal(result.error.name, 'TypeError');
  } finally { delete globalThis[fixtureKey]; }
  void fixture;
}
console.log('generation cancellation rejection and device recovery fixture: ok');
