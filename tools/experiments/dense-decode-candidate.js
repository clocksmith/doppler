import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { getKernelPathMatmulVariant } from '../../src/config/kernel-path-loader.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { selectRuleValue } from '../../src/rules/rule-registry.js';

export function buildDenseDecodeCandidate(kernelPath, bindings, capabilities) {
  if (!kernelPath || typeof kernelPath !== 'object' || !Array.isArray(bindings)) {
    throw new Error('Candidate requires an inline kernel path and explicit materialized projection bindings.');
  }
  for (const key of ['hasF16', 'hasSubgroups', 'hasSubgroupId']) {
    if (typeof capabilities?.[key] !== 'boolean') throw new Error(`Candidate requires boolean capability ${key}.`);
  }
  const normalizedCapabilities = Object.fromEntries(
    ['hasF16', 'hasSubgroups', 'hasSubgroupId'].map(key => [key, capabilities[key]])
  );
  if (kernelPath.layerOverrides?.length) {
    throw new Error('Expand layer overrides and specialize their materialized bindings separately before building a shared candidate.');
  }
  const candidate = structuredClone(kernelPath);
  const decisions = [];
  const seen = new Set();
  for (const binding of bindings) {
    const { op, section, M, N, K, activationDtype, weightDtype, outputDtype, transposeB } = binding;
    if (!['layer', 'postLayer'].includes(section) || typeof op !== 'string' || !op) {
      throw new Error('Projection binding requires an explicit layer/postLayer section and operation.');
    }
    const key = `${section}:${op}`;
    if (seen.has(key)) throw new Error(`Duplicate materialized projection binding ${key}.`);
    seen.add(key);
    for (const value of [M, N, K]) {
      if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${key} requires positive integer M/N/K.`);
    }
    if (!['f16', 'f32'].includes(activationDtype) || !['f16', 'f32'].includes(weightDtype)
        || !['f16', 'f32'].includes(outputDtype) || typeof transposeB !== 'boolean') {
      throw new Error(`${key} requires explicit dense storage, activation, output dtype and transpose layout.`);
    }
    const steps = section === 'layer' ? candidate.decode?.steps : candidate.postLayer;
    const matches = steps?.filter(step => step.op === op) ?? [];
    if (matches.length !== 1) throw new Error(`${key} requires exactly one declared decode step.`);
    const step = matches[0];
    const sourceVariant = getKernelPathMatmulVariant(op, 'decode', binding.layerIdx ?? 0, kernelPath);
    const source = getKernelConfig('matmul', sourceVariant);
    if (source.weightDtype !== weightDtype || source.outputDtype !== outputDtype) {
      throw new Error(`${key} declared kernel does not match actual materialized weights/output; resolve correctness before specialization.`);
    }
    for (const [field, actual] of [['inputDtype', activationDtype], ['weightDtype', weightDtype], ['outputDtype', outputDtype]]) {
      if (step.precision?.[field] != null && step.precision[field] !== actual) {
        throw new Error(`${key} explicit ${field} differs from its materialized binding.`);
      }
    }
    const selected = selectRuleValue('kernels', 'matmul', 'denseDecodeCandidate', {
      M, sourceVariant, activationDtype, weightDtype, outputDtype, transposeB, ...normalizedCapabilities,
    });
    const decision = { op, section, M, N, K, activationDtype, weightDtype, outputDtype, transposeB,
      sourceVariant, selectedVariant: selected.variant, reason: selected.reason, qualified: false };
    decisions.push(decision);
    if (selected.variant === null) continue;
    const config = getKernelConfig('matmul', selected.variant);
    if (config.weightDtype !== weightDtype || config.outputDtype !== outputDtype) {
      throw new Error(`${key} candidate changed the weight/output precision contract.`);
    }
    step.kernel = config.shaderFile;
    step.entry = config.entryPoint;
    // Source entry specialization is not valid for the replacement program.
    delete step.constants;
    const digest = KERNEL_REF_CONTENT_DIGESTS[`${step.kernel}#${step.entry}`];
    if (!digest) throw new Error(`${key} selected program has no canonical digest.`);
    step.digest = `sha256:${digest}`;
    decision.program = { kernel: step.kernel, entry: step.entry, digest: step.digest, constants: config.wgslOverrides };
  }
  return { schema: 'doppler.dense-decode-candidate/v1', qualified: false,
    scope: 'Explicit unchanged-weight screening candidate; requires numerical and performance qualification.',
    capabilities: normalizedCapabilities, kernelPath: candidate, decisions };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (process.argv.length !== 4) throw new Error('Usage: node dense-decode-candidate.js input.json output.json');
  const input = JSON.parse(await readFile(process.argv[2], 'utf8'));
  const result = buildDenseDecodeCandidate(input.kernelPath, input.bindings, input.capabilities);
  await writeFile(process.argv[3], JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
}
