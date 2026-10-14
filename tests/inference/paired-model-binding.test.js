import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseManifest } from '../../src/formats/rdrr/parsing.js';
import { bindPairedModel, readFrozenReferenceManifest } from '../helpers/paired-model-binding.js';
const baseline = readFrozenReferenceManifest();
const graph = JSON.parse(readFileSync(new URL('../../artifacts/chunked-prefill-capsule-20261014/source-execution.json', import.meta.url)));
const candidate = structuredClone(baseline); candidate.inference.execution = graph;
assert.doesNotThrow(() => parseManifest(JSON.stringify(candidate)), 'The public parser must admit declared hybrid projection roles.');
assert.equal(bindPairedModel(candidate, baseline, graph).weightsTokenizerAndSourceFactsUnchanged, true);
for (const alter of [m => {m.architecture.hiddenSize++;}, m=>{m.inference.normalization.rmsNormEps=1e-5;},
 m=>{m.inference.session.kvcache.kvDtype='f32';},m=>{m.inference.execution.decode.pop();}]) {
 const changed=structuredClone(candidate);alter(changed);assert.throws(()=>bindPairedModel(changed,baseline,graph));
}
console.log('paired model binding: only exact approved graph delta admitted; geometry/epsilon/cache changes reject');
