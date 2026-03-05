/**
 * Minimal chat example for @simulatte/doppler.
 *
 * Run in browser with an importmap pointing "@simulatte/doppler" to your
 * local or CDN build, or bundle with any ESM-aware toolchain.
 */

import { DopplerProvider } from '@simulatte/doppler';

const MODEL_ID = 'gemma-3-270m-it-wq4k-ef16';
const MODEL_URL =
  'https://huggingface.co/Clocksmith/rdrr/resolve/HEAD/models/gemma-3-270m-it-wq4k-ef16';

async function main() {
  // 1. Initialize WebGPU device and storage.
  await DopplerProvider.init();

  // 2. Load model (downloads shards to OPFS on first run, cached after).
  await DopplerProvider.loadModel(MODEL_ID, MODEL_URL, (progress) => {
    console.log(`[${progress.stage}] ${progress.message}`);
  });

  // 3. Stream a response token by token.
  const pipeline = DopplerProvider.getPipeline();
  const prompt = 'Explain WebGPU in one sentence.';

  process.stdout?.write?.('> ') ?? document.body.append('> ');
  for await (const token of pipeline.generate(prompt, {
    maxTokens: 128,
    temperature: 0.7,
  })) {
    process.stdout?.write?.(token) ?? document.body.append(token);
  }
  console.log('\n');

  // 4. Cleanup.
  await DopplerProvider.destroy();
}

main().catch(console.error);
