import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { auditEntryPointLowerability } from '../../src/converter/execution-candidate-forge.js';
import { materializeSemanticManifestCandidate } from '../../src/converter/semantic-manifest-lowering.js';

async function readJson(file) {
  return JSON.parse(await fs.readFile(file, 'utf8'));
}

const recipe = await readJson('src/config/forge/semantic-lowering/glimmer-30b-text.json');
const sourceReceipt = await readJson(recipe.modelIRReceipt);
const template = await readJson(recipe.template);
const vocabulary = await readJson('src/config/forge/lowering-vocabularies/heterogeneous-text-v2.json');
const checkedInReceipt = await readJson(recipe.receiptOutput);
const checkedInConfig = await readJson(recipe.output);

const receipt = materializeSemanticManifestCandidate({
  modelIR: sourceReceipt.modelIR,
  template,
  recipe,
});
assert.deepEqual(receipt, checkedInReceipt, 'semantic lowering receipt must be deterministic');
assert.deepEqual(receipt.conversionConfig, checkedInConfig, 'semantic lowering config must be deterministic');

const sourceText = sourceReceipt.modelIR.entryPoints.find((entryPoint) => entryPoint.id === 'text.generate');
const packText = receipt.modelIR.entryPoints.find((entryPoint) => entryPoint.id === 'text.generate');
const packVision = receipt.modelIR.entryPoints.find((entryPoint) => entryPoint.id === 'vision.encode');
assert.equal(sourceText.status, 'unlowered', 'source truth must not be rewritten by lowering');
assert.equal(packText.status, 'lowered');
assert.deepEqual(packText.phases, ['prefill', 'decode']);
assert.equal(packVision.status, 'unlowered');
assert.equal(receipt.modelIR.supportScope.sourceTopology, 'complete');
assert.deepEqual(receipt.modelIR.supportScope.loweredEntryPoints, ['text.generate']);
assert.deepEqual(receipt.modelIR.supportScope.qualifiedEntryPoints, []);
assert.deepEqual(receipt.unresolvedFacts, []);

const { inference, session } = receipt.conversionConfig;
assert.equal(inference.attention.queryPreAttnScalar, 128);
assert.equal(inference.attention.queryScale, 3.87);
assert.deepEqual(inference.attention.queryKeyNormWeightLayers, []);
assert.equal(inference.attention.attentionOutputGate, true);
assert.equal(inference.attention.outputGateType, 'sigmoid');
assert.equal(inference.normalization.rmsNormEps, 1e-5);
assert.equal(inference.normalization.rmsNormWeightOffset, true);
assert.equal(inference.normalization.postNormEps, 1e-8);
assert.equal(inference.normalization.postNormWeightOffset, true);
assert.deepEqual(inference.rope.disabledLayers, Array.from({ length: 13 }, (_, index) => (index * 4) + 3));
assert.equal(inference.output.embeddingNormalization.withScale, false);
assert.equal(inference.output.logitOutputScale, 0.19611613513818404);
assert.equal(inference.output.finalLogitSoftcapping, 20);
assert.equal(session.kvcache.maxSeqLen, 131072);
assert.equal(session.speculation.mode, 'none');
assert.equal(session.useSandwichRMSNormPairFusion, false);
assert.equal(session.usePostFfnNextInputRMSNormPairFusion, false);
assert.equal(session.usePostAttnNormFusedGateUp, false);
assert.deepEqual(
  receipt.conversionConfig.execution.prefill.find((step) => step[0] === 'attention'),
  ['attention', 'attn_stream'],
  'Forge must select a prefill kernel compatible with the source-proven 128-wide heads'
);
assert.deepEqual(receipt.conversionConfig.manifest.conversion, {
  convertedAt: '2026-08-23T08:59:59.000Z',
  tool: 'doppler.semantic-manifest-lowering/v1',
});
assert.deepEqual(receipt.conversionConfig.manifest.eosTokenId, [200001, 200008]);
assert.ok(
  receipt.dispositions.some((item) => (
    item.kind === 'source-generation-defaults'
    && item.disposition === 'preserved-with-qualified-override'
  )),
  'source generation defaults must remain explicit when qualification selects greedy decoding'
);

assert.ok(
  receipt.dispositions.some((item) => item.kind === 'kernel-digest-binding' && item.changed),
  'template kernel drift must be disclosed rather than silently normalized'
);
assert.ok(
  receipt.dispositions.some((item) => item.kind === 'conservative-session-policy'),
  'unqualified session policy must have an explicit disposition'
);
assert.ok(
  receipt.dispositions.some((item) => (
    item.kind === 'prefill-attention-mechanism-binding'
    && item.headDim === 128
    && item.templateKernelId === 'attn_head256'
    && item.kernelId === 'attn_stream'
    && item.changed === true
  )),
  'geometry-driven mechanism selection must disclose replacement of an incompatible template specialization'
);

const lowerability = auditEntryPointLowerability({
  modelIR: sourceReceipt.modelIR,
  entryPointId: recipe.entryPointId,
  vocabulary,
});
assert.equal(lowerability.lowerable, true, 'source semantics must fit the generic vocabulary');
assert.equal(lowerability.entryPointStatus, 'unlowered', 'representability must not imply promotion');

const wrongScale = structuredClone(sourceReceipt.modelIR);
wrongScale.blockClasses[0].geometry.queryScale = 4;
assert.throws(
  () => materializeSemanticManifestCandidate({ modelIR: wrongScale, template, recipe }),
  /attention geometry across text block classes/,
  'divergent per-block geometry must fail closed'
);

const wrongOutput = structuredClone(sourceReceipt.modelIR);
wrongOutput.outputHeads[0].properties.operationOrder = ['language-model-head', 'tanh'];
assert.throws(
  () => materializeSemanticManifestCandidate({ modelIR: wrongOutput, template, recipe }),
  /logit operation order/,
  'unsupported output semantics must fail closed'
);

const incompleteStopContract = structuredClone(sourceReceipt.modelIR);
incompleteStopContract.entryPoints.find((entryPoint) => entryPoint.id === 'text.generate')
  .sourceDefaults.stopTokenIds = [200008];
assert.throws(
  () => materializeSemanticManifestCandidate({ modelIR: incompleteStopContract, template, recipe }),
  /must include the component EOS token/,
  'generation termination must preserve the primary model EOS token'
);

const wrongPostNorm = structuredClone(sourceReceipt.modelIR);
wrongPostNorm.blockClasses.forEach((blockClass) => {
  blockClass.normalization.postNormPosition = 'after-residual';
});
assert.throws(
  () => materializeSemanticManifestCandidate({ modelIR: wrongPostNorm, template, recipe }),
  /post-normalization position/,
  'unsupported residual ordering must fail closed'
);

const lowererSource = await fs.readFile('src/converter/semantic-manifest-lowering.js', 'utf8');
assert.doesNotMatch(lowererSource, /glimmer|qwen/i, 'generic lowering must not contain model-family names');

console.log('✔ semantic-manifest-lowering.test.js passed');
