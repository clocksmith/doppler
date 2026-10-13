import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { buildWgslClosure } from '../../../src/tooling/program-bundle/wgsl-closure.js';
import { expandExecutionV1 } from '../../../src/config/schema/execution-v1.schema.js';
import { assertPairedExecutionMechanismClosure } from '../../../src/config/execution-registry-contract.js';

const candidate = '/var/tmp/doppler-resident-preparation-candidate-20261012';
const runtime = candidate + '/consumer/node_modules/doppler-gpu';
const archive = candidate + '/combined-package/doppler-gpu-0.6.28.tgz';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const graphBytes = await fs.readFile(new URL('execution.json', import.meta.url));
const graph = JSON.parse(graphBytes);
const archiveSha256 = sha(await fs.readFile(archive));
assert.equal(archiveSha256, '0fadac5113d2cc154f65d9d40de50722a16454995e927c7d22bedc5f1a4925cd');
assertPairedExecutionMechanismClosure(graph);
const closure = await buildWgslClosure(graph, expandExecutionV1(graph), { repoRoot: runtime });
const priorGraph = JSON.parse(await fs.readFile(new URL('../execution.json', import.meta.url)));
for (const section of ['preLayer', 'prefill', 'decode', 'postLayer']) assert.deepEqual(graph[section], priorGraph[section]);
for (const [key, declaration] of Object.entries(priorGraph.kernels)) assert.deepEqual(graph.kernels[key], declaration);
const { compileExecutionV1 } = await import(pathToFileURL(runtime + '/src/inference/pipelines/text/execution-v1.js'));
const { compileExecutionPlanState } = await import(pathToFileURL(runtime + '/src/inference/pipelines/text/execution-plan.js'));
const { createDopplerConfig } = await import(pathToFileURL(runtime + '/src/config/schema/index.js'));
const manifestPath = '/var/tmp/qwen35-capsule-0fad-20261012/source/capsule-input/model/manifest.json';
const priorManifest = JSON.parse(await fs.readFile(manifestPath));
const manifest = structuredClone(priorManifest);
manifest.inference.execution = graph;
const runtimeConfig = createDopplerConfig().runtime;
const report = JSON.parse(await fs.readFile(new URL('../../resident-pipeline-preparation-20261012/native-qualification/direction-0/partition-0.json', import.meta.url)));
Object.assign(runtimeConfig.inference, structuredClone(report.initialExecutionIdentity.programLoadPolicy.runtimeConfig.inference));
// CPU plan admission needs a concrete generation bound; no tokens are generated.
runtimeConfig.inference.generation.maxTokens = 8;
const compiled = compileExecutionV1({ manifestInference: manifest.inference, modelId: manifest.modelId,
  numLayers: manifest.architecture.numLayers, weightDtype: 'q4k', runtimeSession: runtimeConfig.inference.session,
  kernelPathPolicy: runtimeConfig.inference.kernelPathPolicy });
const plan = compileExecutionPlanState({ runtimeConfig, manifest,
  resolvedKernelPath: compiled.runtimeInferencePatch.kernelPath, kernelPathSource: 'execution-v1' });
assert.equal(plan.primaryPlan.activationPrecision, 'two-f32');
const files = [];
for (const module of closure.modules) {
  const frozen = await fs.readFile(path.join(runtime, 'src/gpu/kernels', module.file));
  const current = await fs.readFile(new URL('../../../src/gpu/kernels/' + module.file, import.meta.url));
  assert.deepEqual(frozen, current, module.file);
  files.push({ file: module.file, entry: module.entryPoint ?? module.entry, sha256: sha(frozen) });
}
const result = { scope: 'CPU declaration, frozen runtime admission and exact shader-source verification; no physical qualification',
  archive, archiveSha256, runtime, graphSha256: sha(graphBytes), priorGraphSha256: sha(await fs.readFile(new URL('../execution.json', import.meta.url))),
  priorManifestPath: manifestPath, priorManifestSha256: sha(await fs.readFile(manifestPath)),
  semanticStepsUnchanged: true, previousKernelDeclarationsUnchanged: true, shaderBytesUnchanged: true,
  frozenRuntimePlanPrecision: plan.primaryPlan.activationPrecision, sourceFiles: files, physicalQualification: false };
await fs.writeFile(new URL('frozen-runtime-verification.json', import.meta.url), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ graphSha256: result.graphSha256, modules: files.length, frozenRuntimeAdmission: true, physicalQualification: false }));
