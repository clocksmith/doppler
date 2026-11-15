import { isGgufBlockDtype } from "../../../../formats/gguf/block-storage.js";
import { kernelPathRequiresWeightDtype } from "../../../../config/kernel-path-loader.js";

function normalizeExecutionLayerType(layerType) {
  return typeof layerType === 'string' ? layerType.trim().toLowerCase() : '';
}

function isLinearExecutionLayerType(layerType) {
  const normalized = normalizeExecutionLayerType(layerType);
  return normalized === 'linear_attention'
    || normalized === 'linear'
    || normalized === 'gated_delta'
    || normalized === 'gated_delta_net';
}

function isFullAttentionExecutionLayerType(layerType) {
  const normalized = normalizeExecutionLayerType(layerType);
  return normalized === 'full_attention'
    || normalized === 'full'
    || normalized === 'global'
    || normalized === 'standard';
}

// Linear-attention layers alias q_proj/o_proj: global programs must match packed storage.
function isPackedProjectionKernel(kernelEntry, packedDtype) {
  return kernelEntry != null && kernelPathRequiresWeightDtype(
    { decode: { steps: [kernelEntry] } }, packedDtype, 'matmul'
  );
}

function collectPhaseOpEntries(phaseEntries, kernels, ops) {
  const entries = [];
  for (const entry of phaseEntries ?? []) {
    if (Array.isArray(entry)) {
      if (!ops.has(entry[0])) {
        continue;
      }
      entries.push({
        op: entry[0],
        layers: 'all',
        kernelEntry: kernels?.[entry[1]] ?? null,
      });
      continue;
    }
    if (!entry || typeof entry !== 'object' || !Array.isArray(entry.layers) || !Array.isArray(entry.steps)) {
      continue;
    }
    for (const step of entry.steps) {
      if (!Array.isArray(step) || !ops.has(step[0])) {
        continue;
      }
      entries.push({
        op: step[0],
        layers: entry.layers,
        kernelEntry: kernels?.[step[1]] ?? null,
      });
    }
  }
  return entries;
}

function collectCoveredLayers(entries, targetLayers, predicate) {
  const covered = new Set();
  for (const entry of entries) {
    if (!predicate(entry.kernelEntry)) {
      continue;
    }
    if (entry.layers === 'all') {
      for (const layerIdx of targetLayers) {
        covered.add(layerIdx);
      }
      continue;
    }
    for (const layerIdx of entry.layers ?? []) {
      if (targetLayers.includes(layerIdx)) {
        covered.add(layerIdx);
      }
    }
  }
  return covered;
}

export function assertHybridLinearProjectionIsolation(execution, layerTypes, modelId, weightDtype) {
  if (!Array.isArray(layerTypes) || layerTypes.length === 0) {
    return;
  }
  const normalizedWeightDtype = String(weightDtype || '').trim().toLowerCase();
  if (normalizedWeightDtype === 'f16' || normalizedWeightDtype === 'f32') {
    return;
  }
  const packedDtype = isGgufBlockDtype(normalizedWeightDtype) ? normalizedWeightDtype : 'q4k';

  const linearLayers = layerTypes
    .map((layerType, layerIdx) => ({ layerType, layerIdx }))
    .filter(({ layerType }) => isLinearExecutionLayerType(layerType))
    .map(({ layerIdx }) => layerIdx);
  const fullAttentionLayers = layerTypes
    .map((layerType, layerIdx) => ({ layerType, layerIdx }))
    .filter(({ layerType }) => isFullAttentionExecutionLayerType(layerType))
    .map(({ layerIdx }) => layerIdx);

  if (linearLayers.length === 0 || fullAttentionLayers.length === 0) {
    return;
  }

  const guardedOps = [
    {
      phase: 'decode',
      label: 'q_proj/linear_qkv_proj',
      ops: new Set(['q_proj', 'qkv_proj', 'linear_qkv_proj']),
    },
    {
      phase: 'decode',
      label: 'o_proj/linear_out_proj',
      ops: new Set(['o_proj', 'linear_out_proj']),
    },
    {
      phase: 'prefill',
      label: 'q_proj/linear_qkv_proj',
      ops: new Set(['q_proj', 'qkv_proj', 'linear_qkv_proj']),
    },
    {
      phase: 'prefill',
      label: 'o_proj/linear_out_proj',
      ops: new Set(['o_proj', 'linear_out_proj']),
    },
  ];

  for (const guard of guardedOps) {
    const entries = collectPhaseOpEntries(execution?.[guard.phase], execution?.kernels, guard.ops);
    if (entries.length === 0) {
      continue;
    }

    const hasUnsafeGlobalEntry = entries.some(
      (entry) => entry.layers === 'all' && !isPackedProjectionKernel(entry.kernelEntry, packedDtype)
    );
    if (!hasUnsafeGlobalEntry) {
      continue;
    }

    const coveredLinearLayers = collectCoveredLayers(
      entries,
      linearLayers,
      (kernelEntry) => isPackedProjectionKernel(kernelEntry, packedDtype)
    );
    const missingLinearLayers = linearLayers.filter((layerIdx) => !coveredLinearLayers.has(layerIdx));
    if (missingLinearLayers.length === 0) {
      continue;
    }

    throw new Error(
      `[ExecutionV1] Hybrid linear-attention model "${modelId}" cannot apply a global non-Q4 ` +
      `${guard.phase} ${guard.label} kernel because linear-attention layers alias that role. ` +
      `Missing declared Q4 coverage for linear layers [${missingLinearLayers.join(', ')}]. ` +
      'Keep the linear-attention path on registered Q4 weight kernels or isolate those layers with explicit execution-v1 overrides.'
    );
  }
}
