#!/usr/bin/env node
// Explicit offline candidate preparation. Never loads a model or promotes a lane.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import { buildInlineKernelPath } from '../../src/inference/pipelines/text/execution-runtime-builders.js';
import { getLayerSteps, getKernelPathStepConfig } from '../../src/config/kernel-path-loader.js';
import { resolveGroupedPrefill } from '../../src/inference/pipelines/text/moe/grouped-prefill.js';
import { parseManifest } from '../../src/formats/rdrr/index.js';
import { buildWgslClosure } from '../../src/tooling/program-bundle/wgsl-closure.js';

const GROUPED_VARIANTS = Object.freeze({
  moe_grouped_sort: 'native_f32',
  moe_grouped_gu: 'native_f32_ordered',
  moe_grouped_down: 'native_f32_ordered',
});
const GROUPED_OPS = Object.keys(GROUPED_VARIANTS);
const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const ungrouped = steps => steps.filter(step => !GROUPED_OPS.includes(step.op));

export function orderedGroupedDeclarations() {
  return Object.fromEntries(Object.entries(GROUPED_VARIANTS).map(([op, variant]) => {
    const config = getKernelConfig(op, variant);
    const digest = KERNEL_REF_CONTENT_DIGESTS[`${config.shaderFile}#${config.entryPoint}`];
    assert.match(digest ?? '', /^[0-9a-f]{64}$/, `Registered shader digest required: ${op}/${variant}`);
    return [op, { kernel: config.shaderFile, entry: config.entryPoint, digest: `sha256:${digest}` }];
  }));
}

export function buildOrderedGroupedProgram(baseline, numLayers) {
  assert(Number.isSafeInteger(numLayers) && numLayers > 0, 'Explicit positive layer count required');
  assert.equal(baseline?.activationDtype, 'f32', 'Ordered grouped projection requires declared F32 activation');
  assert(baseline.decode?.steps?.length && baseline.prefill?.steps?.length,
    'Complete, separately declared decode and prefill programs required');
  const declarations = orderedGroupedDeclarations();
  const firstDecode = baseline.decode.steps[0];
  const fragment = buildInlineKernelPath(expandExecutionV1({
    kernels: { ...declarations, preserved_decode: firstDecode },
    prefill: GROUPED_OPS.map(op => [op, op]),
    decode: [[firstDecode.op, 'preserved_decode']],
  }), { compute: { defaults: { activationDtype: baseline.activationDtype } },
    kvcache: { kvDtype: baseline.kvDtype } }, 'ordered-grouped-fragment', numLayers);
  const additions = fragment.prefill.steps.map(step => ({ ...step, digest: declarations[step.op].digest }));

  function replace(steps) {
    const declared = steps.filter(step => GROUPED_OPS.includes(step.op));
    if (declared.length) {
      // Use production admission rather than accepting partial or foreign groups.
      resolveGroupedPrefill({ kernelPath: { prefill: { steps } }, phase: 'prefill', layerIdx: 0 });
      return steps.map(step => GROUPED_OPS.includes(step.op)
        ? { ...step, ...declarations[step.op] } : structuredClone(step));
    }
    return [...structuredClone(steps), ...structuredClone(additions)];
  }

  const candidate = structuredClone(baseline);
  candidate.prefill.steps = replace(baseline.prefill.steps);
  for (const override of candidate.layerOverrides ?? []) {
    assert(Array.isArray(override.layers) && override.layers.length > 0
      && override.layers.every(layer => Number.isSafeInteger(layer) && layer >= 0 && layer < numLayers),
    'Layer override scope must fit the supplied layer count');
    if (override.prefill?.steps?.length) override.prefill.steps = replace(override.prefill.steps);
    else if (override.steps?.length) {
      // Shared overrides keep their decode meaning; only prefill gets the addition.
      override.prefill = { ...override.prefill, steps: replace(override.steps) };
    }
  }
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    assert.equal(resolveGroupedPrefill({ kernelPath: baseline, phase: 'decode', layerIdx }), false);
    assert.equal(resolveGroupedPrefill({ kernelPath: candidate, phase: 'prefill', layerIdx }), true);
    assert.deepEqual(getLayerSteps(candidate, layerIdx, 'decode'), getLayerSteps(baseline, layerIdx, 'decode'),
      `Decode changed at layer ${layerIdx}`);
    assert.deepEqual(ungrouped(getLayerSteps(candidate, layerIdx, 'prefill')),
      ungrouped(getLayerSteps(baseline, layerIdx, 'prefill')), `Non-grouped prefill changed at layer ${layerIdx}`);
  }
  return candidate;
}

export async function prepareOrderedGroupedCandidate(manifest, runtimeConfig, options = {}) {
  // Parsing and closure hashing are the production owners; no private parser.
  parseManifest(JSON.stringify(manifest));
  const candidateManifest = structuredClone(manifest);
  const candidateRuntime = structuredClone(runtimeConfig);
  candidateRuntime.inference.kernelPath = buildOrderedGroupedProgram(runtimeConfig.inference.kernelPath,
    manifest.architecture.numLayers);
  const execution = candidateManifest.inference.execution;
  const declarations = orderedGroupedDeclarations();
  for (const [op, declaration] of Object.entries(declarations)) {
    const id = `${op}_ordered_candidate`;
    if (Object.hasOwn(execution.kernels, id)) assert.deepEqual(execution.kernels[id], declaration,
      `Existing candidate mechanism differs: ${id}`);
    execution.kernels[id] = declaration;
    execution.mechanismKernels ??= [];
    if (!execution.mechanismKernels.includes(id)) execution.mechanismKernels.push(id);
  }
  parseManifest(JSON.stringify(candidateManifest));
  const closure = await buildWgslClosure(execution, expandExecutionV1(execution), {
    repoRoot: options.repoRoot ?? repositoryRoot,
  });
  const routing = Array.from({ length: manifest.architecture.numLayers }, (_, layerIdx) => ({
    layerIdx,
    projections: getLayerSteps(candidateRuntime.inference.kernelPath, layerIdx, 'prefill')
      .filter(step => GROUPED_OPS.includes(step.op)).map(step => {
        const config = getKernelPathStepConfig(step.op, step, 'prefill');
        return { op: step.op, variant: config.variant, ...declarations[step.op],
          constants: config.wgslOverrides };
      }),
  }));
  return { manifest: candidateManifest, runtimeConfig: candidateRuntime, closure, routing };
}

async function main() {
  const [manifestPath, configPath, outputDir, ...extra] = process.argv.slice(2);
  assert(manifestPath && configPath && outputDir && extra.length === 0,
    'Usage: node tools/experiments/moe-grouped-ordered-program.js MANIFEST.json RUNTIME_CONFIG.json NEW_OUTPUT_DIR');
  const manifestText = await readFile(manifestPath, 'utf8');
  const configText = await readFile(configPath, 'utf8');
  const prepared = await prepareOrderedGroupedCandidate(JSON.parse(manifestText), JSON.parse(configText));
  await mkdir(outputDir, { recursive: false });
  const write = async (name, value) => writeFile(join(outputDir, name), JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
  await write('candidate-manifest.json', prepared.manifest);
  await write('runtime-candidate.json', prepared.runtimeConfig);
  await write('closure.json', prepared.closure);
  const receipt = {
    scope: 'Offline opt-in preparation only; no model/device run, promotion or source-artifact mutation',
    inputs: { manifestPath: resolve(manifestPath), manifestSha256: sha256(manifestText),
      configPath: resolve(configPath), configSha256: sha256(configText) },
    routing: prepared.routing,
    gates: { productionManifestParser: true, productionWgslClosure: true,
      productionGroupedAdmission: true, decodePreserved: true, nonGroupedPrefillPreserved: true },
    qualification: { componentBitwiseGpu: 'pending', longFullModelAbsoluteTolerance: 0.001,
      longFullModelRelativeTolerance: 0, longFullModel: 'pending', speed: 'unmeasured' },
  };
  await write('preparation.json', receipt);
  console.log(JSON.stringify({ outputDir: resolve(outputDir), ...receipt }));
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  await main();
}
