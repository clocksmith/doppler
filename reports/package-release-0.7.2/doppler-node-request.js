// Execute a frozen package with exact token IDs from the shared workload.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import os from 'node:os';
const [packageDir, modelDir, contractFile, out] = process.argv.slice(2);
if (!out) throw new Error('Usage: node doppler-node-request.js PACKAGE MODEL CONTRACT OUT');
const contract = JSON.parse(await fs.readFile(contractFile, 'utf8'));
const report = { schema: 'doppler.node-request-comparison/v1', startedAt: new Date().toISOString(),
  host: { hostname: os.hostname(), platform: os.platform(), totalMemoryBytes: os.totalmem() },
  contract, modelDir, packageDir, samples: [] };
let model;
try {
  const runtime = await import(pathToFileURL(path.resolve(packageDir, 'src/index.js')).href);
  report.version = runtime.DOPPLER_VERSION;
  const start = performance.now();
  model = await runtime.dr.load({ url: pathToFileURL(path.resolve(modelDir)).href }, { cache: false });
  report.loadMs = performance.now() - start;
  console.log(JSON.stringify({ loaded: true, loadMs: report.loadMs }));
  for (let i = 0; i < contract.warmupRuns + contract.runs; i++) {
    const begin = performance.now();
    const evidence = await model.generateWithEvidence(contract.promptRendered, {
      inputIds: contract.inputTokenIds, useChatTemplate: false,
      maxTokens: contract.maxTokens, temperature: 0, topK: 1, topP: 1,
      repetitionPenalty: 1, seed: 0,
    });
    const sample = { warmup: i < contract.warmupRuns, completeRequestMs: performance.now() - begin, evidence };
    report.samples.push(sample);
    await fs.writeFile(out, JSON.stringify(report, null, 2));
    console.log(JSON.stringify({ sample: i, wallMs: sample.completeRequestMs, tokens: evidence.tokenIds.length, text: evidence.outputText }));
    if (evidence.tokenIds.length !== contract.maxTokens) throw new Error('Generation did not perform requested token work');
  }
} catch (error) { report.error = String(error.stack ?? error); process.exitCode = 1; }
finally {
  await model?.unload();
  const { releaseNodeWebGPU } = await import(pathToFileURL(path.resolve(packageDir, 'src/tooling/node-webgpu.js')).href);
  await releaseNodeWebGPU();
  await fs.writeFile(out, `${JSON.stringify(report, null, 2)}\n`);
}
