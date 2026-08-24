import { getKernelCapabilities } from '../../../gpu/device.js';
import { selectRuleValue as selectKernelRuleValue } from '../../../gpu/kernels/rule-registry.js';
import { trace as debugTrace } from '../../../debug/index.js';
import { getRuntimeConfig } from '../../../config/runtime.js';
import {
  kernelPathRequiresBF16Weights,
  kernelPathRequiresF32MatmulWeights,
} from '../../../config/kernel-path-loader.js';
import { resolveCapabilityTransforms } from '../../../config/transforms/capability-transform-resolver.js';

const Q4K_LAYOUT_ALLOWLIST = new Set(['row', 'col']);
const Q4K_PROJECTION_OPS = new Set([
  'q_proj', 'k_proj', 'v_proj', 'o_proj',
  'gate_proj', 'up_proj', 'down_proj',
  'qkv_proj',
  'linear_qkv_proj',
  'linear_z_proj',
  'linear_a_proj',
  'linear_b_proj',
  'linear_out_proj',
]);
const Q4K_MATMUL_ROLE_OPS = new Map([
  ['q_proj', 'q_proj'],
  ['k_proj', 'k_proj'],
  ['v_proj', 'v_proj'],
  ['o_proj', 'o_proj'],
  ['gate_proj', 'ffn_gate'],
  ['up_proj', 'ffn_up'],
  ['down_proj', 'ffn_down'],
  ['qkv_proj', 'qkv_proj'],
  ['linear_qkv_proj', 'linear_qkv_proj'],
  ['linear_z_proj', 'linear_z_proj'],
  ['linear_a_proj', 'linear_a_proj'],
  ['linear_b_proj', 'linear_b_proj'],
  ['linear_out_proj', 'linear_out_proj'],
  ['lm_head', 'lm_head'],
]);

function summarizeQ4KProjectionKernelKinds(kernelPath) {
  const summary = {
    denseProjectionKernels: [],
    fusedProjectionKernels: [],
  };
  if (!kernelPath || typeof kernelPath !== 'object') {
    return summary;
  }

  const appendPhase = (phase, steps) => {
    for (const step of steps ?? []) {
      const op = step?.op;
      if (!Q4K_PROJECTION_OPS.has(op)) continue;
      const kernel = String(step?.kernel ?? '');
      if (!kernel) continue;
      const descriptor = `${phase}.${op}:${kernel}#${String(step?.entry ?? 'main')}`;
      if (kernel.startsWith('fused_matmul_q4')) {
        summary.fusedProjectionKernels.push(descriptor);
      } else if (kernel.startsWith('matmul_')) {
        summary.denseProjectionKernels.push(descriptor);
      }
    }
  };

  appendPhase('decode', kernelPath.decode?.steps);
  appendPhase('prefill', kernelPath.prefill?.steps);
  return summary;
}

function collectQ4KFusedRoles(kernelPath) {
  const roles = new Set();
  if (!kernelPath || typeof kernelPath !== 'object') return [];

  const appendSteps = (steps) => {
    for (const step of steps ?? []) {
      const role = Q4K_MATMUL_ROLE_OPS.get(step?.op);
      if (!role) continue;
      const kernel = String(step?.kernel ?? '');
      if (kernel.startsWith('fused_matmul_q4')) roles.add(role);
    }
  };

  appendSteps(kernelPath.decode?.steps);
  appendSteps(kernelPath.prefill?.steps);
  appendSteps(kernelPath.postLayer);
  return [...roles].sort();
}

function resolveQ4KProjectionMaterializationMode(manifest, kernelPath, kernelPathSource) {
  const summary = summarizeQ4KProjectionKernelKinds(kernelPath);
  const mode = selectKernelRuleValue('matmul', 'q4kMaterializationMode', {
    hasFusedProjections: summary.fusedProjectionKernels.length > 0,
    hasDenseProjections: summary.denseProjectionKernels.length > 0,
  });
  debugTrace.loader(
    `Q4K materialization: model=${manifest?.modelId ?? 'unknown'}, mode=${mode}, ` +
    `source=${kernelPathSource}, dense=${summary.denseProjectionKernels.length}, ` +
    `fused=${summary.fusedProjectionKernels.length}`
  );
  return mode;
}

function isRetainQ4KMaterializationDisabledByCapability(manifest, caps) {
  if (!caps) return false;
  const adapterInfo = caps.adapterInfo ?? {};
  const platform = {
    id: adapterInfo.device ?? 'unknown',
    vendor: adapterInfo.vendor ?? 'unknown',
    architecture: adapterInfo.architecture ?? 'unknown',
  };
  const runtimeSession = getRuntimeConfig().inference?.session ?? {};
  const graphContext = {
    modelId: manifest?.modelId ?? 'unknown',
    activationDtype: runtimeSession.compute?.defaults?.activationDtype ?? null,
    kvDtype: runtimeSession.kvcache?.kvDtype ?? null,
    retainQ4KMaterialization: true,
  };
  const resolved = resolveCapabilityTransforms(caps, platform, graphContext);
  return resolved.names.includes('disableRetainQ4KMaterialization');
}

export function resolveWeightLoadingConfig(
  manifest,
  kernelPath,
  kernelPathSource = 'none',
  keepF32Weights = false
) {
  const caps = getKernelCapabilities();
  const hasSubgroups = caps != null && caps.hasSubgroups === true;
  const q4kLayout = manifest?.quantizationInfo?.layout ?? null;
  const isQ4KModel = manifest?.quantization === 'Q4_K_M';
  const q4kFusedRoles = isQ4KModel ? collectQ4KFusedRoles(kernelPath) : [];
  if (isQ4KModel && q4kLayout == null) {
    throw new Error(
      `Manifest "${manifest?.modelId ?? 'unknown'}" is missing quantizationInfo.layout for Q4_K_M. Re-convert the model.`
    );
  }
  if (q4kLayout != null && !Q4K_LAYOUT_ALLOWLIST.has(q4kLayout)) {
    throw new Error(
      `Manifest "${manifest?.modelId ?? 'unknown'}" has invalid quantizationInfo.layout "${q4kLayout}". ` +
      `Allowed values: ${[...Q4K_LAYOUT_ALLOWLIST].join(', ')}.`
    );
  }

  let q4kMaterializationMode = isQ4KModel
    ? resolveQ4KProjectionMaterializationMode(manifest, kernelPath, kernelPathSource)
    : 'dense';
  if (isQ4KModel && hasSubgroups && q4kMaterializationMode === 'dense') {
    const runtimeRetain = getRuntimeConfig().inference?.session?.retainQ4KMaterialization === true;
    if (runtimeRetain) {
      if (isRetainQ4KMaterializationDisabledByCapability(manifest, caps)) {
        debugTrace.loader(
          'Q4K materialization retain request disabled by capability rule: ' +
          `model=${manifest?.modelId ?? 'unknown'}, adapter=${caps.adapterInfo?.vendor ?? 'unknown'}/` +
          `${caps.adapterInfo?.architecture ?? 'unknown'}, mode=${q4kMaterializationMode}`
        );
      } else {
        q4kMaterializationMode = 'mixed';
        debugTrace.loader('Q4K materialization overridden by runtime flag retainQ4KMaterialization=true: mode=mixed');
      }
    }
  }
  if (isQ4KModel) {
    debugTrace.loader(
      `Q4K projection materialization: model=${manifest?.modelId ?? 'unknown'}, ` +
      `mode=${q4kMaterializationMode}, source=${kernelPathSource}`
    );
  }

  const hasExplicitKernelPath = kernelPath != null;
  let useFused = isQ4KModel && (
    hasExplicitKernelPath ? q4kMaterializationMode !== 'dense' : hasSubgroups
  );
  if (q4kLayout === 'col') useFused = false;
  const resolvedKeepF32Weights = keepF32Weights || kernelPathRequiresF32MatmulWeights(kernelPath);
  const keepBF16Weights = kernelPathRequiresBF16Weights(kernelPath);
  const pathLabel = kernelPath?.id ?? 'auto';
  const layoutLabel = q4kLayout ?? 'none';
  debugTrace.loader(
    `Weight loading config: fusedQ4K=${useFused}, kernelPath=${pathLabel}, source=${kernelPathSource}, ` +
    `layout=${layoutLabel}, q4kMaterialization=${q4kMaterializationMode}, ` +
    `keepF32Weights=${resolvedKeepF32Weights}, keepBF16Weights=${keepBF16Weights}, ` +
    `subgroups=${hasSubgroups}`
  );

  return {
    useFusedQ4K: useFused,
    q4kLayout,
    keepF32Weights: resolvedKeepF32Weights,
    keepBF16Weights,
    q4kMaterializationMode,
    q4kFusedRoles,
  };
}

export const resolveQ4KConfig = resolveWeightLoadingConfig;
