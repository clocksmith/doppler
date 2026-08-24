import path from 'node:path';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../config/kernels/kernel-ref-digests.js';
import { sha256Hex } from '../../formats/sha256.js';
import {
  createPackageSourceFile,
  hashStableJson,
  normalizeSlash,
  tryReadTextFile,
} from './materialize.js';
import { normalizeDigest, requireString } from './validation.js';

function collectKernelRefsFromEntries(entries, section, refs) {
  if (!Array.isArray(entries)) return;
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    if (Array.isArray(entry)) {
      if (typeof entry[1] === 'string' && entry[1].trim()) {
        refs.push({ id: entry[1].trim(), section, index });
      }
    } else if (entry && typeof entry === 'object' && Array.isArray(entry.steps)) {
      collectKernelRefsFromEntries(entry.steps, section, refs);
    }
  }
}

function collectReachableKernelRefs(execution) {
  const refs = [];
  collectKernelRefsFromEntries(execution.preLayer, 'preLayer', refs);
  collectKernelRefsFromEntries(execution.decode, 'decode', refs);
  collectKernelRefsFromEntries(execution.prefill, 'prefill', refs);
  collectKernelRefsFromEntries(execution.postLayer, 'postLayer', refs);
  for (const [index, id] of (execution.mechanismKernels ?? []).entries()) {
    refs.push({ id, section: 'mechanismKernels', index });
  }
  return refs;
}

function countExpandedStepsByPhase(expandedSteps) {
  const phases = { prefill: 0, decode: 0, preLayer: 0, postLayer: 0 };
  for (const step of expandedSteps) {
    if (step.section === 'preLayer') phases.preLayer += 1;
    else if (step.section === 'postLayer') phases.postLayer += 1;
    else if (step.phase === 'prefill') phases.prefill += 1;
    else if (step.phase === 'decode') phases.decode += 1;
  }
  return phases;
}

function buildKernelIdLookup(execution, modules = []) {
  const lookup = new Map();
  const reachableIds = new Set(modules.map((module) => module.id));
  for (const [id, declaration] of Object.entries(execution.kernels || {})) {
    const key = `${declaration.kernel}#${declaration.entry}#${normalizeDigest(
      declaration.digest,
      `execution.kernels.${id}.digest`
    )}`;
    const current = lookup.get(key);
    if (!current || (!reachableIds.has(current) && reachableIds.has(id))) lookup.set(key, id);
  }
  return lookup;
}

function normalizeLayersForStep(layers) {
  if (layers === 'all') return 'all';
  return Array.isArray(layers) ? layers : [];
}

export function buildExecutionStepMetadata(execution, expandedSteps, modules) {
  const kernelIdLookup = buildKernelIdLookup(execution, modules);
  const moduleById = new Map(modules.map((module) => [module.id, module]));
  const steps = expandedSteps.map((step, index) => {
    const digest = normalizeDigest(step.digest, `expandedSteps[${index}].digest`);
    const kernelId = kernelIdLookup.get(`${step.kernel}#${step.entry}#${digest}`);
    if (!kernelId) {
      throw new Error(
        `program bundle export: expanded step ${index} ${step.op} cannot be mapped to a declared kernel id.`
      );
    }
    const bindings = Array.isArray(moduleById.get(kernelId)?.metadata?.bindings)
      ? moduleById.get(kernelId).metadata.bindings.map((binding) => ({
        group: binding.group,
        binding: binding.binding,
        name: binding.name,
        addressSpace: binding.addressSpace,
        access: binding.access,
      }))
      : [];
    return {
      id: `${step.section}_${step.phase}_${index}_${step.op}`,
      index,
      op: step.op,
      phase: step.phase,
      section: step.section,
      layers: normalizeLayersForStep(step.layers),
      src: typeof step.src === 'string' && step.src.trim() ? step.src : 'state',
      dst: typeof step.dst === 'string' && step.dst.trim() ? step.dst : 'state',
      kernelId,
      kernel: step.kernel,
      entry: step.entry,
      kernelDigest: digest,
      weights: step.weights ?? null,
      constants: step.constants ?? null,
      precision: step.precision ?? null,
      dispatch: {
        phase: step.phase,
        workgroups: `symbolic:${step.section}:${step.phase}:${step.op}`,
        bindings,
      },
    };
  });
  return { steps, stepMetadataHash: hashStableJson(steps) };
}

async function resolveKernelSourceDigest(kernel, entry, kernelSourceRoot) {
  const registryDigest = KERNEL_REF_CONTENT_DIGESTS[`${kernel}#${entry}`];
  const kernelPath = path.resolve(kernelSourceRoot, kernel);
  const source = await tryReadTextFile(kernelPath);
  if (registryDigest) {
    return {
      digest: `sha256:${registryDigest}`,
      sourcePath: normalizeSlash(path.join(kernelSourceRoot, kernel)),
      sourceText: source,
    };
  }
  if (source != null) {
    const normalizedSource = source.replace(/\r\n/g, '\n');
    return {
      digest: `sha256:${sha256Hex(`${normalizedSource}\n@@entry:${entry}`)}`,
      sourcePath: normalizeSlash(path.relative(process.cwd(), kernelPath)),
      sourceText: normalizedSource,
    };
  }
  return { digest: null, sourcePath: null, sourceText: null };
}

function parseWgslBindings(sourceText) {
  if (typeof sourceText !== 'string') return [];
  const bindings = [];
  const pattern = /@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<([^>]+)>)?\s+([A-Za-z_]\w*)/g;
  let match;
  while ((match = pattern.exec(sourceText)) !== null) {
    const address = typeof match[3] === 'string'
      ? match[3].split(',').map((part) => part.trim()).filter(Boolean)
      : [];
    bindings.push({
      group: Number(match[1]),
      binding: Number(match[2]),
      addressSpace: address[0] ?? null,
      access: address[1] ?? null,
      name: match[4],
    });
  }
  return bindings.sort((left, right) => (
    left.group - right.group
      || left.binding - right.binding
      || left.name.localeCompare(right.name)
  ));
}

function parseWgslOverrides(sourceText) {
  if (typeof sourceText !== 'string') return [];
  const overrides = [];
  const pattern = /\boverride\s+([A-Za-z_]\w*)(?:\s*:\s*([^=;]+))?(?:\s*=\s*([^;]+))?;/g;
  let match;
  while ((match = pattern.exec(sourceText)) !== null) {
    overrides.push({
      name: match[1],
      type: typeof match[2] === 'string' ? match[2].trim() : null,
      defaultValue: typeof match[3] === 'string' ? match[3].trim() : null,
    });
  }
  return overrides.sort((left, right) => left.name.localeCompare(right.name));
}

function buildWgslMetadata(sourceText, entry) {
  const workgroupMatch = typeof sourceText === 'string'
    ? /@workgroup_size\(([^)]*)\)/.exec(sourceText)
    : null;
  const metadata = {
    entry,
    bindings: parseWgslBindings(sourceText),
    overrides: parseWgslOverrides(sourceText),
    workgroupSize: workgroupMatch
      ? workgroupMatch[1].split(',').map((part) => part.trim()).filter(Boolean)
      : [],
    requiresSubgroups: typeof sourceText === 'string' && /\b(subgroup|enable\s+subgroups)\b/.test(sourceText),
  };
  return { ...metadata, sourceMetadataHash: hashStableJson(metadata) };
}

export async function buildWgslClosure(execution, expandedSteps, options) {
  const declaredKernelIds = Object.keys(execution.kernels || {}).sort();
  const reachableRefs = collectReachableKernelRefs(execution);
  const reachableKernelIds = [...new Set(reachableRefs.map((ref) => ref.id))].sort();
  const undeclaredKernelRefs = reachableRefs
    .filter((ref) => !execution.kernels?.[ref.id])
    .map((ref) => `${ref.section}[${ref.index}]:${ref.id}`);
  if (undeclaredKernelRefs.length > 0) {
    throw new Error(
      `program bundle export: execution graph references undeclared kernels: ${undeclaredKernelRefs.join(', ')}.`
    );
  }

  const kernelSourceRoot = path.resolve(options.repoRoot, options.kernelSourceRoot || 'src/gpu/kernels');
  const modules = [];
  const packageFiles = [];
  for (const id of reachableKernelIds) {
    const declaration = execution.kernels[id];
    const file = requireString(declaration.kernel, `execution.kernels.${id}.kernel`);
    const entry = requireString(declaration.entry, `execution.kernels.${id}.entry`);
    const declaredDigest = normalizeDigest(declaration.digest, `execution.kernels.${id}.digest`);
    const sourceDigest = await resolveKernelSourceDigest(file, entry, kernelSourceRoot);
    if (sourceDigest.digest && sourceDigest.digest !== declaredDigest) {
      throw new Error(
        `program bundle export: kernel digest mismatch for ${file}#${entry}. `
        + `execution declares ${declaredDigest}, source has ${sourceDigest.digest}.`
      );
    }
    if (sourceDigest.sourceText == null) {
      throw new Error(
        `program bundle export: reachable kernel source ${file}#${entry} is unavailable for closed packaging.`
      );
    }
    const packageFile = createPackageSourceFile({
      role: 'wgsl-source',
      id,
      extension: 'wgsl',
      source: sourceDigest.sourceText,
    });
    packageFiles.push(packageFile);
    modules.push({
      id,
      file,
      entry,
      digest: declaredDigest,
      sourcePath: packageFile.path,
      sourceHash: packageFile.hash,
      reachable: true,
      metadata: buildWgslMetadata(sourceDigest.sourceText, entry),
    });
  }
  return {
    modules,
    packageFiles,
    kernelClosure: {
      declaredKernelIds,
      reachableKernelIds,
      excludedKernelIds: declaredKernelIds.filter((id) => !reachableKernelIds.includes(id)),
      undeclaredKernelRefs,
      expandedStepCount: expandedSteps.length,
      phases: countExpandedStepsByPhase(expandedSteps),
    },
  };
}
