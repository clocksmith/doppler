import assert from 'node:assert/strict';

import {
  classifyTensor,
  classifyTensorRole,
  getGroupType,
  parseGroupLayerIndex,
  parseGroupExpertIndex,
  sortGroupIds,
} from '../../src/formats/rdrr/classification.js';

import {
  getExpectedShardHash,
  parseTensorMap,
} from '../../src/formats/rdrr/parsing.js';

// ============================================================================
// classifyTensor: embedding patterns
// ============================================================================

{
  assert.equal(classifyTensor('model.embed_tokens.weight', 'transformer'), 'embed');
  assert.equal(classifyTensor('token_embd.weight', 'transformer'), 'embed');
  assert.equal(classifyTensor('transformer.wte.weight', 'transformer'), 'embed');
  assert.equal(classifyTensor('model.word_embeddings.weight', 'transformer'), 'embed');
}

// ============================================================================
// classifyTensor: head patterns
// ============================================================================

{
  assert.equal(classifyTensor('lm_head.weight', 'transformer'), 'head');
  assert.equal(classifyTensor('output.weight', 'transformer'), 'head');
  assert.equal(classifyTensor('model.norm.weight', 'transformer'), 'head');
  assert.equal(classifyTensor('model.final_norm.weight', 'transformer'), 'head');
  assert.equal(classifyTensor('norm_f.weight', 'transformer'), 'head');
}

// ============================================================================
// classifyTensor: layer extraction
// ============================================================================

{
  assert.equal(classifyTensor('model.layers.0.self_attn.q_proj.weight', 'transformer'), 'layer.0');
  assert.equal(classifyTensor('model.layers.15.mlp.gate_proj.weight', 'transformer'), 'layer.15');
  assert.equal(classifyTensor('model.layer_42.ffn.weight', 'transformer'), 'layer.42');
}

// ============================================================================
// classifyTensor: no layer match returns 'other'
// ============================================================================

{
  assert.equal(classifyTensor('some_random_tensor.weight', 'transformer'), 'other');
}

// ============================================================================
// classifyTensor: MoE expert extraction
// ============================================================================

{
  assert.equal(
    classifyTensor('model.layers.5.block_sparse_moe.experts.3.w1.weight', 'transformer'),
    'layer.5.expert.3'
  );
  assert.equal(
    classifyTensor('model.layers.0.expert_0.gate_proj.weight', 'transformer'),
    'layer.0.expert.0'
  );
}

// ============================================================================
// classifyTensor: shared expert (no digit after "expert" => falls through)
// ============================================================================
// shared_expert does NOT match /experts?[._](\d+)/ because there's no digit
// after "expert.", so it classifies as a regular layer.

{
  assert.equal(
    classifyTensor('model.layers.2.shared_expert.w1.weight', 'transformer'),
    'layer.2'
  );
  // But shared_expert WITH a digit index returns the shared_expert classification
  assert.equal(
    classifyTensor('model.layers.2.shared_expert_0.w1.weight', 'transformer'),
    'layer.2.shared_expert'
  );
}

// ============================================================================
// classifyTensor: MoE router
// ============================================================================

{
  assert.equal(
    classifyTensor('model.layers.0.block_sparse_moe.gate.weight', 'transformer'),
    'layer.0.shared'
  );
  assert.equal(
    classifyTensor('model.layers.3.router.weight', 'transformer'),
    'layer.3.shared'
  );
}

// ============================================================================
// classifyTensor: diffusion model types
// ============================================================================

{
  assert.equal(classifyTensor('text_encoder.layer.0.weight', 'diffusion'), 'text_encoder');
  assert.equal(classifyTensor('text_encoder_2.layer.0.weight', 'diffusion'), 'text_encoder_2');
  assert.equal(classifyTensor('vae.decoder.weight', 'diffusion'), 'vae');
  assert.equal(classifyTensor('transformer.blocks.0.weight', 'diffusion'), 'transformer');
  assert.equal(classifyTensor('unet.down_blocks.0.weight', 'diffusion'), 'transformer');
  assert.equal(classifyTensor('mmdit.blocks.0.weight', 'diffusion'), 'transformer');
  assert.equal(classifyTensor('unknown_component.weight', 'diffusion'), 'other');
}

// ============================================================================
// classifyTensor: hybrid model (jamba)
// ============================================================================

{
  assert.equal(
    classifyTensor('model.layers.0.self_attn.q_proj.weight', 'jamba'),
    'layer.0.attn'
  );
  assert.equal(
    classifyTensor('model.layers.0.mamba.conv1d.weight', 'jamba'),
    'layer.0.mamba'
  );
}

// ============================================================================
// classifyTensor: pure mamba/rwkv
// ============================================================================

{
  assert.equal(classifyTensor('model.layers.3.mixer.weight', 'mamba'), 'layer.3');
  assert.equal(classifyTensor('model.layers.1.channel_mixing.weight', 'rwkv'), 'layer.1');
}

// ============================================================================
// classifyTensorRole: various roles
// ============================================================================

{
  assert.equal(classifyTensorRole('model.embed_tokens.weight'), 'embedding');
  assert.equal(classifyTensorRole('token_embd.weight'), 'embedding');
  assert.equal(classifyTensorRole('wte.weight'), 'embedding');
  assert.equal(classifyTensorRole('lm_head.weight'), 'lm_head');
  assert.equal(classifyTensorRole('output.weight'), 'lm_head');

  // Matmul weights
  assert.equal(classifyTensorRole('model.layers.0.self_attn.q_proj.weight'), 'matmul');
  assert.equal(classifyTensorRole('model.layers.0.mlp.gate_proj.weight'), 'matmul');
  assert.equal(classifyTensorRole('model.layers.0.mlp.down_proj.weight'), 'matmul');
  assert.equal(classifyTensorRole('model.layers.0.self_attn.o_proj.weight'), 'matmul');

  // Norm weights
  assert.equal(classifyTensorRole('model.layers.0.input_layernorm.weight'), 'norm');
  assert.equal(classifyTensorRole('model.layers.0.ln_1.weight'), 'norm');

  // Expert/router
  assert.equal(classifyTensorRole('model.layers.0.experts.0.w1.weight'), 'expert');
  assert.equal(classifyTensorRole('model.layers.0.shared_expert.gate_proj.weight'), 'expert');
  assert.equal(classifyTensorRole('model.layers.0.router.weight'), 'router');
  assert.equal(classifyTensorRole('model.layers.0.block_sparse_moe.gate.weight'), 'router');
}

// ============================================================================
// classifyTensorRole: diffusion modulation linears are matmul, not norm
// ============================================================================

{
  assert.equal(classifyTensorRole('transformer.blocks.0.norm1.linear.weight'), 'matmul');
  assert.equal(classifyTensorRole('transformer.blocks.0.norm1_context.linear.weight'), 'matmul');
  assert.equal(classifyTensorRole('transformer.blocks.0.norm_out.linear.weight'), 'matmul');
}

// ============================================================================
// classifyTensorRole: unknown returns 'other'
// ============================================================================

{
  assert.equal(classifyTensorRole('some.random.bias'), 'other');
}

// ============================================================================
// classifyTensorRole: attn_output.weight is matmul, not lm_head
// ============================================================================

{
  assert.equal(classifyTensorRole('model.layers.0.self_attn.attn_output.weight'), 'matmul');
}

// ============================================================================
// getGroupType: standard groups
// ============================================================================

{
  assert.equal(getGroupType('embed', 'transformer'), 'embed');
  assert.equal(getGroupType('head', 'transformer'), 'head');
  assert.equal(getGroupType('other', 'transformer'), 'layer');
  assert.equal(getGroupType('layer.0', 'transformer'), 'layer');
  assert.equal(getGroupType('layer.0.expert.3', 'transformer'), 'expert');
  assert.equal(getGroupType('layer.0.shared_expert', 'transformer'), 'shared');
  assert.equal(getGroupType('layer.0.shared', 'transformer'), 'layer');
  assert.equal(getGroupType('layer.0.attn', 'jamba'), 'attn');
  assert.equal(getGroupType('layer.0.mamba', 'jamba'), 'mamba');
}

// ============================================================================
// getGroupType: diffusion groups
// ============================================================================

{
  assert.equal(getGroupType('text_encoder', 'diffusion'), 'text_encoder');
  assert.equal(getGroupType('text_encoder_2', 'diffusion'), 'text_encoder');
  assert.equal(getGroupType('transformer', 'diffusion'), 'transformer');
  assert.equal(getGroupType('unet', 'diffusion'), 'transformer');
  assert.equal(getGroupType('vae', 'diffusion'), 'vae');
}

// ============================================================================
// getGroupType: mamba/rwkv model types
// ============================================================================

{
  assert.equal(getGroupType('layer.0', 'mamba'), 'mamba');
  assert.equal(getGroupType('layer.5', 'rwkv'), 'rwkv');
}

// ============================================================================
// parseGroupLayerIndex
// ============================================================================

{
  assert.equal(parseGroupLayerIndex('layer.0'), 0);
  assert.equal(parseGroupLayerIndex('layer.15'), 15);
  assert.equal(parseGroupLayerIndex('layer.3.expert.2'), 3);
  assert.equal(parseGroupLayerIndex('embed'), undefined);
  assert.equal(parseGroupLayerIndex('head'), undefined);
}

// ============================================================================
// parseGroupExpertIndex
// ============================================================================

{
  assert.equal(parseGroupExpertIndex('layer.0.expert.0'), 0);
  assert.equal(parseGroupExpertIndex('layer.5.expert.7'), 7);
  assert.equal(parseGroupExpertIndex('layer.0'), undefined);
  assert.equal(parseGroupExpertIndex('embed'), undefined);
}

// ============================================================================
// sortGroupIds: ordering contract
// ============================================================================

{
  const ids = ['layer.2', 'head', 'layer.0', 'embed', 'layer.1', 'layer.0.expert.1', 'layer.0.expert.0'];
  const sorted = sortGroupIds(ids);

  assert.equal(sorted[0], 'embed');
  assert.equal(sorted[sorted.length - 1], 'head');

  const layerIndices = sorted.filter(s => s !== 'embed' && s !== 'head')
    .map(s => parseGroupLayerIndex(s));
  for (let i = 1; i < layerIndices.length; i++) {
    assert.ok(layerIndices[i] >= layerIndices[i - 1], 'layers must be sorted by index');
  }

  const expert0Idx = sorted.indexOf('layer.0.expert.0');
  const expert1Idx = sorted.indexOf('layer.0.expert.1');
  assert.ok(expert0Idx < expert1Idx, 'experts must be sorted by expert index');
}

// ============================================================================
// sortGroupIds: diffusion ordering
// ============================================================================

{
  const ids = ['vae', 'transformer', 'text_encoder_2', 'text_encoder'];
  const sorted = sortGroupIds(ids);
  assert.equal(sorted[0], 'text_encoder');
  assert.equal(sorted[1], 'text_encoder_2');
  assert.equal(sorted[2], 'transformer');
  assert.equal(sorted[3], 'vae');
}

// ============================================================================
// sortGroupIds: shared before experts within same layer
// ============================================================================

{
  const ids = ['layer.0.expert.0', 'layer.0.shared', 'layer.0.expert.1'];
  const sorted = sortGroupIds(ids);
  assert.equal(sorted[0], 'layer.0.shared');
}

// ============================================================================
// getExpectedShardHash: sha256 default
// ============================================================================

{
  const shard = { hash: 'abc123', blake3: 'def456' };
  assert.equal(getExpectedShardHash(shard), 'abc123');
  assert.equal(getExpectedShardHash(shard, 'sha256'), 'abc123');
  assert.equal(getExpectedShardHash(shard, 'blake3'), 'def456');
}

// ============================================================================
// getExpectedShardHash: fallback when only one hash present
// ============================================================================

{
  assert.equal(getExpectedShardHash({ hash: 'abc' }), 'abc');
  assert.equal(getExpectedShardHash({ blake3: 'def' }), 'def');
  assert.equal(getExpectedShardHash({ hash: 'abc' }, 'blake3'), 'abc');
  assert.equal(getExpectedShardHash({ blake3: 'def' }, 'sha256'), 'def');
}

// ============================================================================
// getExpectedShardHash: invalid input
// ============================================================================

{
  assert.equal(getExpectedShardHash(null), '');
  assert.equal(getExpectedShardHash(undefined), '');
  assert.equal(getExpectedShardHash('string'), '');
  assert.equal(getExpectedShardHash([1, 2]), '');
}

// ============================================================================
// parseTensorMap: valid tensor map
// ============================================================================

{
  const map = parseTensorMap(JSON.stringify({
    'model.embed_tokens.weight': {
      shardIndex: 0,
      offset: 0,
      size: 1024,
      shape: [32000, 4096],
      role: 'embedding',
    },
    'model.layers.0.q_proj.weight': {
      shard: 1,
      offset: 100,
      size: 512,
      shape: [4096, 4096],
      role: 'matmul',
    },
  }));

  assert.equal(map['model.embed_tokens.weight'].shardIndex, 0);
  assert.equal(map['model.embed_tokens.weight'].shard, 0);
  assert.equal(map['model.embed_tokens.weight'].size, 1024);
  assert.deepEqual(map['model.embed_tokens.weight'].shape, [32000, 4096]);

  assert.equal(map['model.layers.0.q_proj.weight'].shardIndex, 1);
  assert.equal(map['model.layers.0.q_proj.weight'].shard, 1);
}

// ============================================================================
// parseTensorMap: missing required fields throw
// ============================================================================

{
  assert.throws(
    () => parseTensorMap(JSON.stringify({
      'tensor1': { offset: 0, size: 100, shape: [10], role: 'other' },
    })),
    /missing shard index/
  );

  assert.throws(
    () => parseTensorMap(JSON.stringify({
      'tensor1': { shardIndex: 0, size: 100, shape: [10], role: 'other' },
    })),
    /missing offset/
  );

  assert.throws(
    () => parseTensorMap(JSON.stringify({
      'tensor1': { shardIndex: 0, offset: 0, shape: [10], role: 'other' },
    })),
    /missing size/
  );

  assert.throws(
    () => parseTensorMap(JSON.stringify({
      'tensor1': { shardIndex: 0, offset: 0, size: 100, role: 'other' },
    })),
    /missing shape/
  );

  assert.throws(
    () => parseTensorMap(JSON.stringify({
      'tensor1': { shardIndex: 0, offset: 0, size: 100, shape: [10] },
    })),
    /missing role/
  );
}

// ============================================================================
// parseTensorMap: invalid JSON throws
// ============================================================================

{
  assert.throws(
    () => parseTensorMap('not json'),
    /Failed to parse tensors.json/
  );
}

// ============================================================================
// parseTensorMap: spans normalization
// ============================================================================

{
  const map = parseTensorMap(JSON.stringify({
    'tensor1': {
      shardIndex: 0,
      offset: 0,
      size: 200,
      shape: [10, 20],
      role: 'matmul',
      spans: [
        { shardIndex: 0, offset: 0, size: 100 },
        { shard: 1, offset: 0, size: 100 },
      ],
    },
  }));

  assert.equal(map['tensor1'].spans.length, 2);
  assert.equal(map['tensor1'].spans[0].shardIndex, 0);
  assert.equal(map['tensor1'].spans[1].shardIndex, 1);
}

console.log('rdrr-validation.test: ok');
