/**
 * Minimal chat example for @simulatte/doppler.
 *
 * Run in browser with an importmap pointing "@simulatte/doppler" to your
 * local or CDN build, or bundle with any ESM-aware toolchain.
 */

import { doppler } from '@simulatte/doppler';

const MODEL_URL =
  'https://huggingface.co/Clocksmith/rdrr/resolve/HEAD/models/gemma-3-270m-it-wq4k-ef16';

async function main() {
  const model = await doppler.load({ url: MODEL_URL }, {
    onProgress(progress) {
    console.log(`[${progress.stage}] ${progress.message}`);
    },
  });

  const prompt = 'Explain WebGPU in one sentence.';

  process.stdout?.write?.('> ') ?? document.body.append('> ');
  for await (const token of model.generate(prompt, {
    maxTokens: 128,
    temperature: 0.7,
  })) {
    process.stdout?.write?.(token) ?? document.body.append(token);
  }
  console.log('\n');

  await model.unload();
}

main().catch(console.error);
