import assert from 'node:assert/strict';
import { registerPipeline, getPipelineFactory, normalizeModelType } from '../../src/inference/pipelines/registry.js';
import { InferencePipeline } from '../../src/inference/pipelines/text/pipeline.js';

// Model types are case-insensitive identifiers. A manifest that says
// "DiffusionGemma" must reach the same factory as "diffusion_gemma" would
// reach after its lazy module registers under the lower-cased name.
const factory = () => ({ kind: 'test' });
registerPipeline('Registry-Case-Probe', factory);
assert.equal(getPipelineFactory('registry-case-probe'), factory);
assert.equal(getPipelineFactory('REGISTRY-CASE-PROBE'), factory);
assert.equal(getPipelineFactory(' Registry-Case-Probe '), factory);
assert.equal(normalizeModelType('  Mixed_Case '), 'mixed_case');
assert.equal(getPipelineFactory(''), null);
assert.equal(getPipelineFactory(undefined), null);
assert.equal(getPipelineFactory('never-registered'), null);

// Capabilities come from the manifest, not from the presence of a method that
// every text pipeline has.
const capabilities = Object.getOwnPropertyDescriptor(InferencePipeline.prototype, 'capabilities').get;
const describe = (manifest, extra = {}) => capabilities.call({ manifest, prefillWithEmbedding() {}, visionCapable: false, audioCapable: false, ...extra });
assert.deepEqual(describe({ inference: {} }), ['generation']);
assert.deepEqual(describe({ inference: { supportsEmbedding: true } }), ['generation', 'embedding']);
assert.deepEqual(describe({ inference: { supportsRerank: true } }), ['generation', 'rerank']);
assert.deepEqual(describe({ inference: { supportsEmbedding: true, supportsSequence: true } }), ['generation', 'embedding', 'sequence']);
assert.deepEqual(describe({ inference: {} }, { visionCapable: true }), ['generation', 'multimodal', 'video']);
assert.deepEqual(describe({ inference: {} }, { audioCapable: true }), ['generation', 'audio']);
console.log('pipeline registry and capabilities ok');
