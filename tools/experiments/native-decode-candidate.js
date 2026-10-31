import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { getLayerSteps, getKernelPathStepConfig } from '../../src/config/kernel-path-loader.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

// Explicit input selections only. This builder neither promotes nor edits an artifact.
export function buildNativeDecodeCandidate(kernelPath, numLayers, projections, sharedSlotLayers) {
  if (kernelPath?.activationDtype !== 'f32' || !Number.isSafeInteger(numLayers) || numLayers < 1
    || !Array.isArray(projections) || !Array.isArray(sharedSlotLayers)) {
    throw new Error('Native candidate requires F32 activation facts, layer count, projections and sharedSlotLayers.');
  }
  const candidate = structuredClone(kernelPath);
  const layers = Array.from({ length: numLayers }, (_, layerIdx) => ({ layers: [layerIdx],
    decode: { steps: structuredClone(getLayerSteps(kernelPath, layerIdx, 'decode')) },
    prefill: { steps: structuredClone(getLayerSteps(kernelPath, layerIdx, 'prefill')) },
  }));
  const bind = (step, config) => {
    const digest = KERNEL_REF_CONTENT_DIGESTS[`${config.shaderFile}#${config.entryPoint}`];
    if (!digest) throw new Error('Candidate program requires a canonical digest.');
    const replacement = { ...step, kernel: config.shaderFile, entry: config.entryPoint, digest: `sha256:${digest}` };
    delete replacement.constants;
    return replacement;
  };
  const seen = new Set();
  for (const { layerIdx, op, variant } of projections) {
    if (!Number.isSafeInteger(layerIdx) || !layers[layerIdx]) throw new Error('Projection layer is outside declared model.');
    const key = `${layerIdx}:${op}`;
    if (seen.has(key)) throw new Error(`Duplicate projection ${key}.`);
    seen.add(key);
    const steps = layers[layerIdx].decode.steps;
    const matches = steps.filter(step => step.op === op);
    if (matches.length !== 1) throw new Error(`${key} requires one declared projection.`);
    const source = getKernelPathStepConfig('matmul', matches[0], 'decode');
    const target = getKernelConfig('matmul', variant);
    if (!source || target.variantMetadata?.q40Geometry !== 'cooperative'
      || source.weightDtype !== target.weightDtype || target.outputDtype !== 'f32'
      || source.outputDtype !== 'f32') throw new Error(`${key} must preserve actual native storage and F32 output.`);
    for (const field of ['activationDtype', 'inputDtype', 'outputDtype']) {
      if (matches[0].precision?.[field] != null && matches[0].precision[field] !== 'f32') {
        throw new Error(`${key} conflicts with F32 precision.`);
      }
    }
    steps[steps.indexOf(matches[0])] = bind(matches[0], target);
  }
  const sharedSeen = new Set();
  for (const layerIdx of sharedSlotLayers) {
    if (!Number.isSafeInteger(layerIdx) || !layers[layerIdx] || sharedSeen.has(layerIdx)) {
      throw new Error('Shared slot layers must be distinct valid indices.');
    }
    sharedSeen.add(layerIdx);
    const steps = layers[layerIdx].decode.steps;
    if (steps.some(step => ['moe_shared_combine','moe_shared_slot_gu','moe_shared_slot_down'].includes(step.op))) {
      throw new Error('Shared combine is already declared; do not duplicate its arithmetic.');
    }
    for (const operation of ['moe_shared_slot_gu','moe_shared_slot_down']) {
      steps.push(bind({ op: operation }, getKernelConfig(operation, 'native_f32')));
    }
  }
  candidate.layerOverrides = layers;
  return { schema: 'doppler.native-decode-candidate/v1', qualified: false, kernelPath: candidate,
    scope: 'Unchanged native storage candidate; equal-materialization and best-native comparisons remain separate.' };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (process.argv.length !== 4) throw new Error('Usage: node native-decode-candidate.js input.json output.json');
  const input = JSON.parse(await readFile(process.argv[2], 'utf8'));
  await writeFile(process.argv[3], JSON.stringify(buildNativeDecodeCandidate(input.kernelPath,
    input.numLayers, input.projections, input.sharedSlotLayers), null, 2) + '\n', { flag: 'wx' });
}
