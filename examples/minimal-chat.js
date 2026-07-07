/**
 * Minimal chat example for doppler-gpu.
 *
 * Run in browser with an importmap pointing "doppler-gpu" to your
 * local or CDN build, or bundle with any ESM-aware toolchain.
 */

import { dr } from 'doppler-gpu';

const MODEL_ID = 'qwen3-0.8b';

async function main() {
  const model = await dr.load(MODEL_ID, {
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
