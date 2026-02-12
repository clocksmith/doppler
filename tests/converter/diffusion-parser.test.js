import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const {
  detectDiffusionLayout,
  parseDiffusionModel,
} = await import('../../src/converter/parsers/diffusion.js');

{
  const layout = detectDiffusionLayout({
    transformer: ['diffusers', 'Flux2Transformer2DModel'],
    text_encoder: ['transformers', 'Qwen3ForCausalLM'],
    tokenizer: ['transformers', 'Qwen2TokenizerFast'],
    vae: ['diffusers', 'AutoencoderKLFlux2'],
    scheduler: ['diffusers', 'FlowMatchEulerDiscreteScheduler'],
  });
  assert.equal(layout.id, 'flux');
}

{
  const layout = detectDiffusionLayout({
    transformer: ['diffusers', 'SD3Transformer2DModel'],
    text_encoder: ['transformers', 'CLIPTextModel'],
    text_encoder_2: ['transformers', 'CLIPTextModel'],
    text_encoder_3: ['transformers', 'T5EncoderModel'],
    tokenizer: ['transformers', 'CLIPTokenizer'],
    tokenizer_2: ['transformers', 'CLIPTokenizer'],
    tokenizer_3: ['transformers', 'T5TokenizerFast'],
    vae: ['diffusers', 'AutoencoderKL'],
    scheduler: ['diffusers', 'FlowMatchEulerDiscreteScheduler'],
  });
  assert.equal(layout.id, 'sd3');
}

{
  const files = new Set([
    'model_index.json',
    'transformer/config.json',
    'text_encoder/config.json',
    'vae/config.json',
    'scheduler/scheduler_config.json',
    'transformer/diffusion_pytorch_model.safetensors',
    'text_encoder/model.safetensors.index.json',
    'text_encoder/model-00001-of-00002.safetensors',
    'text_encoder/model-00002-of-00002.safetensors',
    'vae/diffusion_pytorch_model.safetensors',
    'tokenizer/vocab.json',
    'tokenizer/merges.txt',
    'tokenizer/tokenizer_config.json',
    'tokenizer/special_tokens_map.json',
    'tokenizer/tokenizer.json',
  ]);

  const parsed = await parseDiffusionModel({
    findExistingSuffix(suffixes) {
      for (const suffix of suffixes || []) {
        if (files.has(suffix)) return suffix;
      }
      return null;
    },
    async readJson(suffix) {
      if (suffix === 'model_index.json') {
        return {
          _class_name: 'Flux2KleinPipeline',
          transformer: ['diffusers', 'Flux2Transformer2DModel'],
          text_encoder: ['transformers', 'Qwen3ForCausalLM'],
          tokenizer: ['transformers', 'Qwen2TokenizerFast'],
          vae: ['diffusers', 'AutoencoderKLFlux2'],
          scheduler: ['diffusers', 'FlowMatchEulerDiscreteScheduler'],
        };
      }
      if (suffix === 'text_encoder/model.safetensors.index.json') {
        return {
          weight_map: {
            'model.layers.0.self_attn.q_proj.weight': 'model-00001-of-00002.safetensors',
            'model.layers.0.self_attn.k_proj.weight': 'model-00002-of-00002.safetensors',
          },
        };
      }
      return {};
    },
    async readText(suffix) {
      return `text:${suffix}`;
    },
    async readBinary(suffix) {
      return new Uint8Array([1, 2, 3]).buffer;
    },
    async parseSingleSafetensors(suffix) {
      return {
        tensors: [{ name: `${suffix}.weight`, shape: [1], dtype: 'F16', size: 2, offset: 0 }],
      };
    },
    async parseShardedSafetensors(indexSuffix, indexJson) {
      return {
        tensors: Object.keys(indexJson.weight_map).map((name, idx) => ({
          name,
          shape: [1],
          dtype: 'F16',
          size: 2,
          offset: idx * 2,
        })),
      };
    },
  });

  assert.equal(parsed.layout, 'flux');
  assert.equal(parsed.architecture, 'diffusion');
  assert.ok(parsed.tensors.some((tensor) => tensor.name.startsWith('text_encoder.')));
  assert.ok(parsed.auxFiles.some((asset) => asset.name === 'tokenizer_vocab.json'));
  assert.ok(parsed.config?.diffusion?.tokenizers?.text_encoder);
}

console.log('diffusion-parser.test: ok');
