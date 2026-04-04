import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { preprocessGemma4Image } from '../../src/inference/pipelines/vision/gemma4.js';

{
  const preprocessed = preprocessGemma4Image(
    Uint8Array.from([255, 0, 128]),
    1,
    1,
    {
      patchSize: 1,
      poolingKernelSize: 1,
      defaultOutputLength: 1,
    }
  );

  assert.ok(Math.abs(preprocessed.patches[0] - 1.0) < 1e-6);
  assert.ok(Math.abs(preprocessed.patches[1] - 0.0) < 1e-6);
  assert.ok(Math.abs(preprocessed.patches[2] - (128 / 255)) < 1e-6);
}

{
  const width = 780;
  const height = 518;
  const pixels = new Uint8Array(width * height * 3);
  const preprocessed = preprocessGemma4Image(
    pixels,
    width,
    height,
    {
      patchSize: 16,
      poolingKernelSize: 3,
      defaultOutputLength: 280,
    }
  );

  assert.equal(preprocessed.outputLength, 260);
}

const textSource = readFileSync(new URL('../../src/inference/pipelines/text.js', import.meta.url), 'utf8');
const generatorSource = readFileSync(new URL('../../src/inference/pipelines/text/generator.js', import.meta.url), 'utf8');
const visionSource = readFileSync(new URL('../../src/inference/pipelines/vision/gemma4.js', import.meta.url), 'utf8');
const attentionSource = readFileSync(new URL('../../src/gpu/kernels/attention_streaming.wgsl', import.meta.url), 'utf8');

assert.match(
  textSource,
  /const imageTokenSpanLength = encodeResult\.numTokens;/,
  'Gemma 4 transcribeImage must expand the image placeholder to the encoder-reported soft-token count'
);

assert.match(
  textSource,
  /__internalEmbeddingInputSpan:\s*\{\s*offset:\s*imageStartOffset,\s*length:\s*encodeResult\.numTokens,\s*tokenId:\s*padTokenId,\s*\}/,
  'Gemma 4 transcribeImage must pass the PAD-token replacement span into prefill'
);

assert.match(
  textSource,
  /__internalMultimodalBidirectionalSpan:\s*\{\s*offset:\s*imageStartOffset,\s*length:\s*encodeResult\.numTokens,\s*\}/,
  'Gemma 4 transcribeImage must preserve the image-token span for bidirectional multimodal prefill attention'
);

assert.match(
  visionSource,
  /numKVHeads,\s*scale:\s*1\.0,\s*causal:\s*false/,
  'Gemma 4 vision attention must override the generic attention scale to 1.0 after Q/K normalization'
);

assert.match(
  visionSource,
  /activatedTensor = await runGeLU\(\s*gateTensor,\s*\{\s*size:\s*numTokens \* intermediateSize,\s*gate:\s*upTensor,\s*\}\s*\);/,
  'Gemma 4 vision MLP must compute GeLU(gate_proj(x)) * up_proj(x) to match the upstream Gemma 4 implementation'
);

assert.match(
  generatorSource,
  /context\.multimodalBidirectionalSpan = multimodalBidirectionalSpan == null\s*\?\s*null\s*:\s*\{\s*start:\s*startPos \+ multimodalBidirectionalSpan\.offset,\s*length:\s*multimodalBidirectionalSpan\.length,\s*\};/,
  'Gemma 4 prefill must convert the multimodal image span into absolute positions before layer attention dispatch'
);

assert.match(
  attentionSource,
  /fn is_bidirectional_span_visible\(abs_query: u32, abs_key: u32\) -> bool \{/,
  'Streaming attention must expose a scoped bidirectional span helper for Gemma 4 multimodal prefill'
);

assert.match(
  attentionSource,
  /if \(u\.is_causal != 0u && abs_key > abs_query\) \{\s*if \(is_bidirectional_span_visible\(abs_query, abs_key\)\) \{ return false; \}\s*return true;\s*\}/,
  'Streaming attention must keep the image-token span bidirectional while preserving causal masking elsewhere'
);

console.log('gemma4-multimodal-prefill-contract.test: ok');
