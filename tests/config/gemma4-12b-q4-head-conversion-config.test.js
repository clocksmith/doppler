import assert from 'node:assert/strict';
import fs from 'node:fs';

import { validateRequiredInferenceFields } from '../../src/inference/pipelines/text/config.js';
import { selectRuleValue } from '../../src/rules/rule-registry.js';

const BASE_MODEL_ID = 'gemma-4-12b-it-text-q4k-ehf16-af16';
const Q4_HEAD_MODEL_ID = 'gemma-4-12b-it-text-q4k-ehf16-hq4k-af16';
const W4A16_QAT_MODEL_ID = 'gemma-4-12b-it-text-w4a16-ct-ehf16-af16';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function phaseStep(steps, op) {
  return steps?.find((step) => Array.isArray(step) && step[0] === op) ?? null;
}

function postLayerStep(execution, op) {
  return phaseStep(execution?.postLayer, op);
}

const baseConfig = readJson(`src/config/conversion/gemma4/${BASE_MODEL_ID}.json`);
const q4HeadConfig = readJson(`src/config/conversion/gemma4/${Q4_HEAD_MODEL_ID}.json`);
const w4a16QatConfig = readJson(`src/config/conversion/gemma4/${W4A16_QAT_MODEL_ID}.json`);

assert.equal(q4HeadConfig.modelType, 'gemma4');
assert.equal(q4HeadConfig.output?.modelBaseId, Q4_HEAD_MODEL_ID);
assert.equal(q4HeadConfig.output?.textOnly, true);
assert.equal(q4HeadConfig.quantization?.weights, 'q4k');
assert.equal(q4HeadConfig.quantization?.embeddings, 'f16');
assert.equal(q4HeadConfig.quantization?.lmHead, 'q4k');
assert.equal(q4HeadConfig.quantization?.computePrecision, 'f16');

assert.equal(q4HeadConfig.inference?.output?.tieWordEmbeddings, true);
assert.equal(q4HeadConfig.manifest?.artifactIdentity?.sourceCheckpointId, 'google/gemma-4-12B-it');
assert.equal(q4HeadConfig.manifest?.artifactIdentity?.sourceRevision, '5926caa4ec0cac5cbfadaf4077420520de1d5205');
assert.equal(
  q4HeadConfig.manifest?.artifactIdentity?.weightPackId,
  `${Q4_HEAD_MODEL_ID}-wp-catalog-v1`
);
assert.equal(
  q4HeadConfig.manifest?.artifactIdentity?.manifestVariantId,
  `${Q4_HEAD_MODEL_ID}-mv-q4-lm-head-f16-activations-v1`
);
assert.equal(q4HeadConfig.manifest?.artifactIdentity?.artifactCompleteness, 'complete');
assert.equal(q4HeadConfig.manifest?.artifactIdentity?.shardSetHash, undefined);
assert.equal(q4HeadConfig.manifest?.weightsRef, undefined);

assert.notEqual(
  q4HeadConfig.manifest?.artifactIdentity?.weightPackId,
  baseConfig.manifest?.artifactIdentity?.weightPackId,
  'Q4-head lane must produce a new weight pack because conversion appends synthetic tied lm_head bytes'
);
assert.equal(baseConfig.quantization?.lmHead, 'f16');
assert.equal(postLayerStep(baseConfig.execution, 'lm_head')?.[1], 'lm_head_gemv_stable');

assert.deepEqual(q4HeadConfig.execution?.kernels?.attn_small, {
  kernel: 'attention_small_f16kv.wgsl',
  entry: 'main',
  digest: 'sha256:ad3bb913c17167eb28fefc6abd24602ea0f148bc22a2b5f437eeffcc6f7fc668',
  precision: {
    activationDtype: 'f32',
    kvDtype: 'f16',
    outputDtype: 'f32',
  },
});
assert.deepEqual(q4HeadConfig.execution?.kernels?.attn_head256, {
  kernel: 'attention_head256_f16kv.wgsl',
  entry: 'main',
  digest: 'sha256:4ecdc079e322a770350414244cb383d72ae7ffa47ed620c64ff425c93413e97a',
  precision: {
    activationDtype: 'f32',
    kvDtype: 'f16',
    outputDtype: 'f32',
  },
});
assert.deepEqual(q4HeadConfig.execution?.kernels?.attn_head512, {
  kernel: 'attention_head512_f16kv.wgsl',
  entry: 'main',
  digest: 'sha256:fe8bc99230a5ee4c969ba1cf9eca289ad936d14f1f87c12a7ca95640cdd8d89e',
  precision: {
    activationDtype: 'f32',
    kvDtype: 'f16',
    outputDtype: 'f32',
  },
});

assert.deepEqual(q4HeadConfig.execution?.kernels?.embed, {
  kernel: 'gather_split4_f16_vec4_f16_out.wgsl',
  entry: 'gather_vec4_f16_out',
  digest: 'sha256:b83e26325355e9270909afaee64b3f21491f853eb9fe72be344d6886a0a4b358',
  precision: {
    inputDtype: 'f16',
    outputDtype: 'f16',
  },
});
assert.deepEqual(q4HeadConfig.execution?.kernels?.rmsnorm, {
  kernel: 'rmsnorm_f16.wgsl',
  entry: 'main',
  digest: 'sha256:5f46959f502a9a1c3a8e7c6bb3ac56dd06d377d9748dbb9957de710f04f6d05b',
});
assert.deepEqual(q4HeadConfig.execution?.kernels?.rope, {
  kernel: 'rope_f16.wgsl',
  entry: 'main',
  digest: 'sha256:8f4769a0dbad218bf4b4631bb097a14ae43a841cbcea1dc3a6f7f6f1a7002f26',
});
assert.deepEqual(q4HeadConfig.execution?.kernels?.gelu, {
  kernel: 'gelu_f16.wgsl',
  entry: 'main',
  digest: 'sha256:9472489a6cd998bce165bacc486c4bbf91320f0aecdf29de9410f715ab791d62',
  constants: {
    HAS_GATE: true,
  },
});
assert.deepEqual(q4HeadConfig.execution?.kernels?.residual, {
  kernel: 'residual_f16.wgsl',
  entry: 'main',
  digest: 'sha256:29c34e627ad251f03a8862fb1f459ab488132ef04192d45e40c78fdf21a34fd5',
});
assert.deepEqual(q4HeadConfig.execution?.kernels?.q4_decode_gemv, {
  kernel: 'fused_matmul_q4_multicol_f16a.wgsl',
  entry: 'main_multicol_f16a',
  digest: 'sha256:bde56c42f6e8a6b4bf5ceaf32b117c6a48c5341801976e049cd48b5f5a768e13',
  precision: {
    inputDtype: 'f16',
    outputDtype: 'f16',
  },
});
assert.deepEqual(q4HeadConfig.execution?.kernels?.q4_decode_gemv_stable, {
  kernel: 'fused_matmul_q4.wgsl',
  entry: 'main_gemv',
  digest: 'sha256:c547743b9806fec47142c6a9f9ac2bd047981fd8cded46ce40dc01725e8aa71d',
  precision: {
    inputDtype: 'f32',
    outputDtype: 'f32',
  },
});
assert.deepEqual(q4HeadConfig.execution?.kernels?.rope_decode_stable, {
  kernel: 'rope.wgsl',
  entry: 'main',
  digest: 'sha256:b2da9d396668981dab9794c2973b668279f768994466b083d2105730555e1a5b',
  precision: {
    inputDtype: 'f32',
    outputDtype: 'f32',
  },
});
assert.deepEqual(q4HeadConfig.execution?.kernels?.attn_decode_stable, {
  kernel: 'attention_decode_online_f16kv.wgsl',
  entry: 'main',
  digest: 'sha256:91808be35895e921c0fa43fc47bfe8cab69c437c388ab561f595d9dc983f66e1',
  precision: {
    activationDtype: 'f32',
    kvDtype: 'f16',
    outputDtype: 'f32',
  },
});
assert.deepEqual(q4HeadConfig.execution?.kernels?.q4_widetile_f16a, {
  kernel: 'fused_matmul_q4_widetile_f16a.wgsl',
  entry: 'main',
  digest: 'sha256:fd0fb3e0a5aeb8189674cfdf450181ca132e4a2db899a122ef93ec0b4abf9b1d',
  precision: {
    inputDtype: 'f16',
    outputDtype: 'f16',
  },
});
assert.deepEqual(q4HeadConfig.execution?.kernels?.fused_ffn_q4k_f16, {
  kernel: 'fused_ffn_q4k_f16.wgsl',
  entry: 'main',
  digest: 'sha256:ec277b54b3635280c9d2a25b4906b131ddcf26936d71ae4d64a0eeddb08efd7a',
  precision: {
    inputDtype: 'f16',
    outputDtype: 'f16',
  },
});

assert.deepEqual(q4HeadConfig.execution?.kernels?.lm_head_q4, {
  kernel: 'fused_matmul_q4.wgsl',
  entry: 'main_gemv',
  digest: 'sha256:c547743b9806fec47142c6a9f9ac2bd047981fd8cded46ce40dc01725e8aa71d',
  constants: {
    COLS_PER_WG: 64,
    THREADS_PER_COL_GEMV: 4,
  },
});
assert.equal(
  postLayerStep(q4HeadConfig.execution, 'lm_head')?.[1],
  'lm_head_q4',
  'Q4-head lane must decode through the packed tied-head kernel'
);
assert.equal(
  postLayerStep(q4HeadConfig.execution, 'lm_head_prefill')?.[1],
  'lm_head_prefill_stable',
  'Q4-head lane should keep the existing stable prefill logits contract until converted hardware evidence says otherwise'
);
assert.equal(phaseStep(q4HeadConfig.execution?.preLayer, 'embed')?.[1], 'embed');
for (const op of ['q_proj', 'k_proj', 'v_proj']) {
  assert.equal(phaseStep(q4HeadConfig.execution?.decode, op)?.[1], 'q4_decode_gemv_stable');
}
for (const op of ['rope_q', 'rope_k']) {
  assert.equal(phaseStep(q4HeadConfig.execution?.decode, op)?.[1], 'rope_decode_stable');
}
assert.equal(phaseStep(q4HeadConfig.execution?.decode, 'attention')?.[1], 'attn_decode_stable');
assert.equal(phaseStep(q4HeadConfig.execution?.decode, 'o_proj')?.[1], 'q4_decode_gemv');
assert.equal(phaseStep(q4HeadConfig.execution?.decode, 'ffn')?.[1], 'fused_ffn_q4k_f16');
for (const op of ['gate_proj', 'up_proj', 'activation', 'down_proj']) {
  assert.equal(phaseStep(q4HeadConfig.execution?.decode, op), null);
}
for (const op of ['o_proj']) {
  assert.equal(phaseStep(q4HeadConfig.execution?.decode, op)?.[1], 'q4_decode_gemv');
}
for (const op of ['attn_residual', 'ffn_residual']) {
  assert.equal(phaseStep(q4HeadConfig.execution?.decode, op)?.[1], 'residual');
}
for (const section of q4HeadConfig.execution?.prefill ?? []) {
  for (const op of ['o_proj', 'down_proj']) {
    assert.equal(phaseStep(section.steps, op)?.[1], 'q4_widetile_f16a');
  }
  for (const op of ['attn_residual', 'ffn_residual']) {
    assert.equal(phaseStep(section.steps, op)?.[1], 'residual');
  }
}
assert.equal(
  selectRuleValue('inference', 'ffn', 'useFusedGateUp', {
    hasGate: true,
    hasUp: true,
    hasDown: true,
    hasFusedWeights: false,
    inputIsSupported: true,
    hasLoRA: false,
    dtypeMatches: true,
    dtypeSupported: true,
    weightDtype: 'q4k',
    activationDtype: 'f16',
    hiddenSizeAligned32: true,
    batchSize: 4,
    useDoubleWideMlp: false,
  }),
  true,
  'Q4K f16 activation decode should use the existing fused gate/up kernel rule'
);

assert.doesNotThrow(
  () => validateRequiredInferenceFields(q4HeadConfig.inference, Q4_HEAD_MODEL_ID),
  'Q4-head conversion config must satisfy required inference fields'
);

assert.equal(w4a16QatConfig.modelType, 'gemma4');
assert.equal(w4a16QatConfig.output?.modelBaseId, W4A16_QAT_MODEL_ID);
assert.equal(w4a16QatConfig.output?.textOnly, true);
assert.equal(w4a16QatConfig.quantization?.weights, 'w4a16');
assert.equal(w4a16QatConfig.quantization?.embeddings, 'f16');
assert.equal(w4a16QatConfig.quantization?.lmHead, 'f16');
assert.equal(w4a16QatConfig.quantization?.sourceTrainingQuantization, 'qat');
assert.equal(w4a16QatConfig.quantization?.sourceQuantizationTarget, 'w4a16');
assert.equal(w4a16QatConfig.quantization?.sourceQuantizationFormat, 'compressed-tensors');
assert.equal(
  w4a16QatConfig.manifest?.artifactIdentity?.sourceCheckpointId,
  'google/gemma-4-12B-it-qat-w4a16-ct'
);
assert.equal(
  w4a16QatConfig.manifest?.artifactIdentity?.sourceRevision,
  'dcfe12254e8fb98c743f21efa05ff64937926c64'
);
assert.equal(
  w4a16QatConfig.manifest?.artifactIdentity?.weightPackId,
  `${W4A16_QAT_MODEL_ID}-wp-catalog-v1`
);
assert.equal(
  w4a16QatConfig.manifest?.artifactIdentity?.manifestVariantId,
  `${W4A16_QAT_MODEL_ID}-mv-w4a16-ct-reference-f16-dequant-v1`
);
assert.equal(w4a16QatConfig.manifest?.artifactIdentity?.artifactCompleteness, 'complete');
assert.equal(w4a16QatConfig.manifest?.weightsRef, undefined);

const qatExecutionText = JSON.stringify(w4a16QatConfig.execution);
assert.equal(qatExecutionText.includes('q4_decode_gemv'), false);
assert.equal(qatExecutionText.includes('q4_widetile'), false);
assert.equal(qatExecutionText.includes('lm_head_q4'), false);
for (const op of ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']) {
  assert.equal(phaseStep(w4a16QatConfig.execution?.decode, op)?.[1], 'gemv');
}
for (const section of w4a16QatConfig.execution?.prefill ?? []) {
  for (const op of ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']) {
    assert.equal(phaseStep(section.steps, op)?.[1], 'tiled');
  }
}
assert.equal(postLayerStep(w4a16QatConfig.execution, 'lm_head')?.[1], 'lm_head_gemv_stable');
assert.equal(postLayerStep(w4a16QatConfig.execution, 'lm_head_prefill')?.[1], 'lm_head_prefill_stable');
assert.doesNotThrow(
  () => validateRequiredInferenceFields(w4a16QatConfig.inference, W4A16_QAT_MODEL_ID),
  'W4A16 QAT conversion config must satisfy required inference fields'
);

console.log('gemma4-12b-q4-head-conversion-config.test: ok');
