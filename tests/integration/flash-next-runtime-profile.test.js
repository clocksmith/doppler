import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolveConversionPlan } from '../../src/converter/conversion-plan.js';
import { extractArchitecture } from '../../src/converter/artifact-identity.js';
import { createManifest } from '../../src/converter/manifest-builder.js';
import { parseModelConfigFromManifest } from '../../src/inference/pipelines/text/config.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { getKernelPathMatmulVariant } from '../../src/config/kernel-path-loader.js';
import { assertHyperConnectionLoad } from '../../src/inference/pipelines/text/hyper-connection/sequence-transaction.js';

// Source metadata and the real conversion recipe meet the runtime boundary.
// No checkpoint weights, fake model outputs or GPU allocations are involved.
const json = async path => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const converterConfig = await json('../../src/config/conversion/qwen3/qwen-3-8-flash-next-text-q4k-ehf16-af32.json');
const rawConfig = await json('../../artifacts/qwen38-flash-next-20261013/source/config.json');
const architecture = extractArchitecture(rawConfig);
const plan = resolveConversionPlan({ rawConfig, tensors: [], converterConfig, architectureConfig: architecture });
const manifest = createManifest(converterConfig.output.modelBaseId, { config: rawConfig, architecture }, [], {}, {
  source: 'hf', modelType: plan.modelType, quantization: plan.manifestQuantization,
  hashAlgorithm: converterConfig.manifest.hashAlgorithm, inference: plan.manifestInference,
  moeConfig: plan.moeConfig, quantizationInfo: plan.quantizationInfo, architecture,
  eosTokenId: converterConfig.manifest.eosTokenId, textOnly: converterConfig.output.textOnly,
});
const modelConfig = parseModelConfigFromManifest(manifest);
assert.equal(modelConfig.visionConfig, null, 'text-only artifact must not request excluded vision weights');
const compiled = compileExecutionV1({ manifestInference: manifest.inference, modelId: manifest.modelId,
  numLayers: architecture.numLayers, headDim: architecture.headDim, weightDtype: plan.manifestQuantization });
const kernelPath = compiled.runtimeInferencePatch.kernelPath;
assert.ok(kernelPath);
assert.equal(compiled.runtimeInferencePatch.pipeline, undefined, 'specialized decoder must bypass the ordinary residual layer pipeline');
const common = ['attn_hc_down', 'attn_hc_up', 'attn_hc_inject', 'mlp_hc_down', 'mlp_hc_up', 'mlp_hc_inject'];
for (const phase of ['prefill', 'decode']) {
  for (let layer = 0; layer < architecture.numLayers; layer++) {
    const roles = [...common];
    if (modelConfig.layerTypes[layer] === 'full_attention') roles.push('qsa_index_qk');
    if (modelConfig.ngramEmbedding.layers.some(spec => spec.decoderLayerIndex === layer)) roles.push('ple_key_proj', 'ple_value_proj');
    for (const role of roles) assert.equal(getKernelPathMatmulVariant(role, phase, layer, kernelPath), 'f32', `${phase} layer ${layer} ${role} retains source control precision`);
  }
  for (const role of ['hc_mixer_down', 'hc_mixer_up']) assert.equal(getKernelPathMatmulVariant(role, phase, 0, kernelPath), 'f32');
}
assert.equal(modelConfig.hyperConnections.streamWidth, architecture.hiddenSize);
assert.equal(modelConfig.indexedAttention.denseVisibleLimit, modelConfig.indexedAttention.tokenBudget + modelConfig.indexedAttention.compressRatio - 1);
assert.deepEqual(modelConfig.ropeInverseFrequencies, converterConfig.inference.rope.ropeInverseFrequencies);
const state = { modelConfig, manifest, useGPU: true, runtimeConfig: { inference: {
  session: compiled.session, compute: { activationDtype: compiled.session.compute.defaults.activationDtype },
} }, executionV1State: compiled };
assert.doesNotThrow(() => assertHyperConnectionLoad(state));
assert.throws(() => assertHyperConnectionLoad({ ...state, modelPartition: { startLayer: 0, endLayer: 4 } }), /unpartitioned/);
console.log('Flash-Next recipe -> conversion -> manifest -> specialized runtime contracts passed (metadata only)');
