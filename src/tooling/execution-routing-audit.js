import { computeCanonicalSha256 } from '../utils/canonical-hash.js';
import { digestRegisteredVariantDescriptor } from './registered-variant-calibration.js';

export const EXECUTION_ROUTING_AUDIT_SCHEMA = 'doppler.execution-routing-audit/v1';

const TILED_VARIANT_PAIRS = Object.freeze({
  f16: 'f16_tiled',
  f16w_f32a: 'f16w_f32a_tiled',
  q4_fused_batched: 'q4_fused_widetile',
  q4_fused_batched_f16a: 'q4_fused_widetile_f16a',
});

const F16_VARIANT_PAIRS = Object.freeze({
  f16w_f32a: 'f16',
  f16w_f32a_tiled: 'f16_tiled',
  gemv: 'gemv_f16a',
  gemv_subgroup: 'gemv_subgroup_f16a',
  gemv_subgroup_vec4: 'gemv_subgroup_vec4_f16a',
  q4_fused_gemv: 'q4_fused_f16a',
  q4_fused_batched: 'q4_fused_batched_f16a',
  q4_fused_widetile: 'q4_fused_widetile_f16a',
});

function collectVariantIndex(registry) {
  const index = new Map();
  for (const [operation, operationConfig] of Object.entries(registry?.operations ?? {})) {
    for (const [variantId, descriptor] of Object.entries(operationConfig?.variants ?? {})) {
      const key = `${descriptor.wgsl}#${descriptor.entryPoint}`;
      const entries = index.get(key) ?? [];
      entries.push({ operation, variantId, descriptor });
      index.set(key, entries);
    }
  }
  return index;
}

function collectPhaseUses(execution) {
  const uses = [];
  for (const phase of ['preLayer', 'prefill', 'decode', 'postLayer']) {
    for (const step of execution?.[phase] ?? []) {
      if (!Array.isArray(step) || typeof step[1] !== 'string') continue;
      uses.push({ phase, role: step[0], kernelId: step[1] });
    }
  }
  return uses;
}

function makeReference(operation, variantId, descriptor, kernelDigests) {
  const digest = kernelDigests[`${descriptor.wgsl}#${descriptor.entryPoint}`];
  return {
    operation,
    variantId,
    descriptorDigest: digestRegisteredVariantDescriptor(operation, variantId, descriptor),
    kernelDigest: digest ? `sha256:${digest}` : null,
    requires: descriptor.requires ?? [],
  };
}

function selectionPolicyForDescriptor(descriptor) {
  const requires = descriptor.requires ?? [];
  return requires.includes('shader-f16')
    || descriptor.inputDtype === 'f16'
    || descriptor.outputDtype === 'f16'
    ? 'required-after-evidence-promotion-on-compatible-hardware'
    : 'best-evidence-wins';
}

function sameKvPrecision(selected, candidate) {
  const selectedUsesF16Kv = selected.variantId.includes('f16kv');
  return selectedUsesF16Kv === candidate.variantId.includes('f16kv');
}

function findOpportunities(manifest, execution, variantIndex, registry, kernelDigests) {
  const opportunities = new Map();
  const headDim = manifest?.architecture?.headDim ?? null;
  const activationDtype = manifest?.inference?.session?.compute?.defaults?.activationDtype ?? null;
  for (const use of collectPhaseUses(execution)) {
    const selectedKernel = execution.kernels?.[use.kernelId];
    if (!selectedKernel) continue;
    const selectedMatches = variantIndex.get(
      `${selectedKernel.kernel}#${selectedKernel.entry}`
    ) ?? [];
    for (const selected of selectedMatches) {
      const variants = Object.entries(registry.operations[selected.operation]?.variants ?? {})
        .map(([variantId, descriptor]) => ({ operation: selected.operation, variantId, descriptor }));
      const proposals = [];
      if (
        selected.operation === 'attention'
        && use.phase === 'prefill'
        && Number.isInteger(headDim)
        && selected.descriptor.variantMetadata?.exactHeadDim !== headDim
      ) {
        proposals.push(...variants.filter((candidate) => (
          candidate.descriptor.variantMetadata?.exactHeadDim === headDim
          && sameKvPrecision(selected, candidate)
        )).map((candidate) => ({
          reason: 'exact-head-prefill-available',
          candidate,
        })));
      }
      if (
        use.phase === 'prefill'
        && selected.operation === 'matmul'
      ) {
        const tiledVariantId = TILED_VARIANT_PAIRS[selected.variantId];
        const candidate = variants.find((entry) => entry.variantId === tiledVariantId);
        if (candidate) {
          proposals.push({ reason: 'tiled-prefill-variant-available', candidate });
        }
      }
      if (activationDtype === 'f32' && selected.descriptor.outputDtype === 'f32') {
        const f16VariantId = selected.operation === 'attention'
          ? selected.variantId.replace(/_f16kv$/, '_f16')
          : F16_VARIANT_PAIRS[selected.variantId];
        const candidate = variants.find((entry) => (
          entry.variantId === f16VariantId && entry.descriptor.outputDtype === 'f16'
        ));
        if (candidate) {
          proposals.push({ reason: 'f16-output-variant-available', candidate });
        }
      }
      for (const proposal of proposals) {
        const key = [
          use.phase,
          use.kernelId,
          selected.operation,
          selected.variantId,
          proposal.candidate.variantId,
          proposal.reason,
        ].join(':');
        const existing = opportunities.get(key);
        if (existing) {
          if (!existing.roles.includes(use.role)) existing.roles.push(use.role);
          continue;
        }
        opportunities.set(key, {
          phase: use.phase,
          roles: [use.role],
          kernelId: use.kernelId,
          reason: proposal.reason,
          selected: makeReference(
            selected.operation,
            selected.variantId,
            selected.descriptor,
            kernelDigests
          ),
          candidate: makeReference(
            proposal.candidate.operation,
            proposal.candidate.variantId,
            proposal.candidate.descriptor,
            kernelDigests
          ),
          disposition: 'calibration-required',
          selectionPolicy: selectionPolicyForDescriptor(proposal.candidate.descriptor),
        });
      }
    }
  }
  return Array.from(opportunities.values()).map((opportunity) => ({
    ...opportunity,
    roles: opportunity.roles.sort(),
  })).sort((left, right) => (
    `${left.phase}:${left.kernelId}:${left.reason}:${left.candidate.variantId}`
      .localeCompare(`${right.phase}:${right.kernelId}:${right.reason}:${right.candidate.variantId}`)
  ));
}

export function auditManifestExecutionRouting(manifest, registry, kernelDigests) {
  const execution = manifest?.inference?.execution ?? null;
  if (!execution?.kernels || typeof execution.kernels !== 'object') {
    throw new Error('execution routing audit: manifest.inference.execution.kernels is required');
  }
  const variantIndex = collectVariantIndex(registry);
  const integrity = Object.entries(execution.kernels).map(([kernelId, reference]) => {
    const key = `${reference.kernel}#${reference.entry}`;
    const expected = kernelDigests[key] ? `sha256:${kernelDigests[key]}` : null;
    return {
      kernelId,
      key,
      declaredDigest: reference.digest ?? null,
      expectedDigest: expected,
      registeredVariants: (variantIndex.get(key) ?? []).map(
        ({ operation, variantId }) => `${operation}/${variantId}`
      ),
      status: expected === null
        ? 'digest-unregistered'
        : reference.digest === expected
          ? 'verified'
          : 'digest-mismatch',
    };
  }).sort((left, right) => left.kernelId.localeCompare(right.kernelId));
  const core = {
    schema: EXECUTION_ROUTING_AUDIT_SCHEMA,
    modelId: manifest.modelId ?? null,
    artifactDigest: manifest?.artifactIdentity?.artifactHash ?? null,
    executionGraphDigest: computeCanonicalSha256(execution),
    integrity,
    opportunities: findOpportunities(
      manifest,
      execution,
      variantIndex,
      registry,
      kernelDigests
    ),
  };
  return { ...core, digest: computeCanonicalSha256(core) };
}
