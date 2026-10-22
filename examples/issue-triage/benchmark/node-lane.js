// One Node benchmark run: opens one lane in this process, decides the issues and
// writes a report shaped like the browser page's. Invoked by node-run.js.
//   node node-lane.js <lane> <run-config.json> <out.json>
// Doppler uses the compat load() path that its Node qualification uses, on the
// Dawn `webgpu` package; Transformers.js uses onnxruntime-node; open-jev keeps its
// own device and dtype choice.
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { TRIAGE_POLICY, documentText } from '../triage.js';
import { OPEN_JEV_QUESTION, rerankerDecider } from './lanes.js';

const [lane, configPath, outputPath] = process.argv.slice(2);
const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
const report = { lane, mode: 'node', quality: config.quality, startedAtMs: performance.now(), platform: { node: process.version } };

async function installWebGpu() {
  const { create, globals } = createRequire(path.join(config.repoRoot, 'package.json'))('webgpu');
  Object.assign(globalThis, globals);
  const gpu = create(process.platform === 'linux' ? ['backend=vulkan'] : []);
  Object.defineProperty(globalThis, 'navigator', { value: { gpu }, configurable: true });
  const adapter = await gpu.requestAdapter();
  const info = adapter?.info ?? {};
  report.gpu = { provider: 'webgpu (Dawn) npm package', adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
    features: adapter ? [...adapter.features].sort() : null };
}

async function openDoppler(modelDirectory, weights) {
  await installWebGpu();
  const api = await import(pathToFileURL(path.join(config.benchmarkRoot, 'node_modules/doppler-gpu/src/index.js')).href);
  const model = await api.load({ url: pathToFileURL(path.resolve(modelDirectory) + '/').href }, { cache: false, isolatedLoader: true });
  return {
    decide: rerankerDecider((request, options) => model.scoreChoices(request, options)),
    runtime: { engine: 'doppler', surface: 'node compat', weights, deviceInfo: model.deviceInfo ?? null },
    close: () => model.unload(),
  };
}

async function transformers() {
  const module = await import(pathToFileURL(path.join(config.benchmarkRoot, 'node_modules/@huggingface/transformers/dist/transformers.node.mjs')).href);
  Object.assign(module.env, { allowRemoteModels: false, allowLocalModels: true, localModelPath: config.hfRoot });
  return module;
}

const OPEN_JEV = { 'open-jev-kev-0.6b': 'kev-0.6b', 'open-jev-kev-4b': 'kev-4b', 'open-jev-deberta': 'open-jev' };
async function openLane() {
  if (lane === 'doppler-q4k') return openDoppler(config.q4kModelDirectory, 'q4k');
  if (lane === 'doppler-f16') return openDoppler(config.f16ModelDirectory, 'f16');
  if (lane === 'transformersjs-fp16' || lane === 'transformersjs-fp16-cpu') {
    const t = await transformers();
    const { openTransformersEngine } = await import('../engines.js');
    // onnxruntime-node bundles a WebGPU backend; the -cpu lane records the CPU path.
    const device = lane === 'transformersjs-fp16' ? 'webgpu' : 'cpu';
    const engine = await openTransformersEngine({ transformers: t, modelId: 'Qwen/Qwen3-Reranker-0.6B', dtype: 'fp16', device });
    return { decide: rerankerDecider(engine.scoreChoices), runtime: { engine: 'transformersjs', version: t.env.version, device, dtype: 'fp16' }, close: engine.close };
  }
  if (OPEN_JEV[lane]) {
    const t = await transformers();
    const { OpenJev, choice } = await import(pathToFileURL(path.join(config.benchmarkRoot, 'node_modules/open-jev/dist/index.js')).href);
    const jev = await OpenJev.load({ model: OPEN_JEV[lane] });
    const question = choice(OPEN_JEV_QUESTION.instructions, OPEN_JEV_QUESTION.options, OPEN_JEV_QUESTION.descriptions);
    return {
      async decide(issue) {
        const [answer] = await jev.decide(documentText(issue, TRIAGE_POLICY.bodyCharacterBudgets[0]), [question]);
        return { probabilities: { ...answer.probabilities }, choice: answer.choice, detail: { confidence: answer.confidence } };
      },
      runtime: { engine: 'open-jev', version: '0.1.2', transformersVersion: t.env.version, ...jev.runtime },
      close: () => jev.dispose(),
    };
  }
  throw new Error(`Unknown Node lane ${lane}`);
}

try {
  const run = JSON.parse(await fs.readFile(config.runConfigPath, 'utf8'));
  const openStart = performance.now();
  const opened = await openLane();
  report.openMs = performance.now() - openStart;
  report.runtime = opened.runtime;
  const decide = async (issue) => {
    const started = performance.now();
    const answer = await opened.decide(issue);
    return { index: issue.index, label: issue.label, repo: issue.repo, elapsedMs: performance.now() - started, ...answer };
  };
  const first = await decide(run.firstIssue);
  report.firstDecisionMs = first.elapsedMs;
  report.coldCompleteMs = performance.now();
  report.firstIssue = first;
  const limit = config.quality ? run.test.length : config.testLimit;
  report.test = [];
  for (const issue of run.test.slice(0, limit)) report.test.push(await decide(issue));
  if (config.quality) {
    report.train = [];
    for (const issue of run.train) report.train.push(await decide(issue));
  }
  await opened.close();
} catch (error) {
  report.error = `${error?.name ?? 'Error'}: ${error?.message ?? error}\n${error?.stack ?? ''}`.slice(0, 2000);
}
report.finishedAtMs = performance.now();
await fs.writeFile(outputPath, `${JSON.stringify(report)}\n`);
process.exit(0);
