import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { extractArchitecture } from '../../src/converter/artifact-identity.js';

import {
  buildRefreshRawConfig,
  extractSourceQuantization,
  rebuildArtifactIdentityForRefresh,
} from '../../tools/refresh-converted-manifest.js';

{
  assert.equal(
    extractSourceQuantization({
      quantizationInfo: {
        weights: 'q4k',
      },
    }),
    'q4k'
  );
  assert.equal(
    extractSourceQuantization({
      quantization: 'F16',
    }),
    'F16'
  );
  assert.equal(extractSourceQuantization({}), '');
}

{
  const rawConfig = buildRefreshRawConfig({
    config: {},
    modelType: 'transformer',
    inference: {
      layerPattern: {
        layerTypes: ['attention', 'ffn'],
      },
    },
  });
  assert.equal(rawConfig.model_type, 'transformer');
  assert.deepEqual(rawConfig.layer_types, ['attention', 'ffn']);
}

{
  // Text-only manifests intentionally omit the raw multimodal source config.
  // An explicit source preserves family semantics instead of inferring them
  // from the generic transformer pipeline kind.
  const source = JSON.parse(await readFile(new URL('../../artifacts/qwen38-flash-next-20261013/source/config.json', import.meta.url)));
  const manifest = { modelType: 'transformer', architecture: extractArchitecture(source) };
  const raw = buildRefreshRawConfig(manifest, source);
  assert.equal(raw.model_type, source.model_type);
  assert.deepEqual(raw.text_config, source.text_config);
  assert.equal(manifest.config, undefined);
  const wrongSource = structuredClone(source);
  wrongSource.text_config.hidden_size++;
  assert.throws(() => buildRefreshRawConfig(manifest, wrongSource), /architecture.hiddenSize disagrees/);
  assert.throws(() => buildRefreshRawConfig(manifest, {}), /non-empty model_type/);
}

{
  const manifest = { modelId: 'model', modelType: 'transformer', hashAlgorithm: 'sha256',
    artifactIdentity: { sourceCheckpointId: 'source', sourceRevision: 'pinned',
      weightCapsuleHash: 'sha256:' + '1'.repeat(64), weightCapsuleId: 'weights',
      shardSetHash: 'sha256:' + '2'.repeat(64), manifestVariantId: 'stale' } };
  const config = { manifest: { artifactIdentity: { sourceCheckpointId: 'source' } } };
  const plan = { modelType: 'transformer', manifestInference: { execution: { kernels: { gate: 'sigmoid' } } } };
  const first = await rebuildArtifactIdentityForRefresh(manifest, config, config, plan, 'recipe.json');
  assert.notEqual(first.manifestVariantId, 'stale');
  assert.equal(first.weightCapsuleHash, manifest.artifactIdentity.weightCapsuleHash);
  assert.equal(first.sourceRevision, 'pinned');
  const changed = structuredClone(plan);
  changed.manifestInference.execution.kernels.gate = 'silu';
  const second = await rebuildArtifactIdentityForRefresh(manifest, config, config, changed, 'recipe.json');
  assert.notEqual(first.manifestVariantId, second.manifestVariantId);
}

{
  const source = {
    model_type: 'lfm2', num_hidden_layers: 16, hidden_size: 2048, intermediate_size: 12288,
    num_attention_heads: 32, num_key_value_heads: 8, vocab_size: 65536, max_position_embeddings: 128000,
    block_auto_adjust_ff_dim: true, block_ffn_dim_multiplier: 1, block_multiple_of: 256,
  };
  const manifest = { modelType: 'transformer', architecture: { ...extractArchitecture(source), intermediateSize: 8192 } };
  assert.deepEqual(buildRefreshRawConfig(manifest, source), source);
  assert.equal(extractArchitecture(source).intermediateSize, 8192);
  assert.equal(extractArchitecture({ ...source, block_ffn_dim_multiplier: 1.05 }).intermediateSize, 8704);
  assert.equal(extractArchitecture({ ...source, intermediate_size: 12300, block_ffn_dim_multiplier: null }).intermediateSize, 8200);
  assert.equal(extractArchitecture({ ...source, block_auto_adjust_ff_dim: false }).intermediateSize, 12288);
  assert.throws(() => buildRefreshRawConfig({ ...manifest, architecture: { ...manifest.architecture, intermediateSize: 7936 } }, source), /architecture.intermediateSize disagrees/);
  assert.throws(() => extractArchitecture({ ...source, block_multiple_of: 0 }), /block_multiple_of/);
  assert.throws(() => extractArchitecture({ ...source, block_ffn_dim_multiplier: -1 }), /block_ffn_dim_multiplier/);
  assert.throws(() => extractArchitecture({ ...source, block_ffn_dim_multiplier: undefined }), /block_ffn_dim_multiplier/);
  assert.throws(() => extractArchitecture({ ...source, intermediate_size: 1, block_ffn_dim_multiplier: null }), /adjusted intermediate_size/);
}

console.log('refresh-converted-manifest.test: ok');
