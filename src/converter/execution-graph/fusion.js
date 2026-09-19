import { F16_TO_F32_ACTIVATION_MAP, F32_TO_F16_ACTIVATION_MAP, KERNEL_FILE_PRECISION_PATCHES, cloneGraph, deriveF16AttentionKernelEntry, deriveKernelEntry, deriveKernelEntryWithPrecision, deriveKernelKey, deriveLinearDecodeF16KernelEntry, deriveLmHeadDecodeF16KernelEntry, deriveQ4DecodeF16KernelEntry, deriveQ4PrefillF16AccumKernelEntry, deriveQ4PrefillF16KernelEntry, deriveQ4WideTilePrefillF16KernelEntry, findPhaseStep, narrowToF16Activations, replacePhaseStepKernelKey, useQwen36F16Activations, useQwenDecodeF16Matmuls, useQwenF16PrimaryMatmuls } from './dtype.js';

export function deriveDenseDecodeF16KernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (base.kernel === 'matmul_gemv_subgroup.wgsl') {
    return {
      ...deriveKernelEntry(base, 'matmul_gemv_subgroup_f16a.wgsl', base.entry ?? 'main'),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  if (base.kernel === 'matmul_gemv_subgroup_f16a.wgsl') {
    return deriveKernelEntryWithPrecision(base, precision);
  }
  return null;
}

export const LAYER_PROJECTION_OPS = new Set([
  'q_proj', 'k_proj', 'v_proj', 'o_proj',
  'gate_proj', 'up_proj', 'down_proj',
]);

export function useGemma4Int4PleSelectiveF16Decode(graph, ctx) {
  const result = cloneGraph(graph);
  let changed = false;

  const decodeProjectionStep = (result.decode || []).find((entry) => Array.isArray(entry) && entry[0] === 'q_proj');
  const decodeProjectionKernel = decodeProjectionStep ? result.kernels[decodeProjectionStep[1]] : null;
  const decodeProjectionEntry = deriveDenseDecodeF16KernelEntry(decodeProjectionKernel);
  if (decodeProjectionStep && decodeProjectionEntry) {
    const decodeProjectionKey = deriveKernelKey(result.kernels, decodeProjectionStep[1], '_gemma4_f16');
    result.kernels[decodeProjectionKey] = decodeProjectionEntry;
    for (const op of ['q_proj', 'k_proj', 'v_proj']) {
      const phaseResult = replacePhaseStepKernelKey(result.decode, op, decodeProjectionKey);
      if (phaseResult.changed) {
        result.decode = phaseResult.steps;
        changed = true;
      }
    }
  }

  for (const op of ['rope_q', 'rope_k']) {
    const step = (result.decode || []).find((entry) => Array.isArray(entry) && entry[0] === op);
    const kernelKey = step?.[1];
    const kernelEntry = kernelKey ? result.kernels[kernelKey] : null;
    const replacement = kernelEntry ? F32_TO_F16_ACTIVATION_MAP.get(kernelEntry.kernel) : null;
    if (!step || !kernelKey || !replacement) {
      continue;
    }
    const ropeKey = deriveKernelKey(result.kernels, kernelKey, '_gemma4_f16');
    result.kernels[ropeKey] = deriveKernelEntryWithPrecision(
      deriveKernelEntry(kernelEntry, replacement, kernelEntry.entry),
      { inputDtype: 'f16', outputDtype: 'f16' }
    );
    const phaseResult = replacePhaseStepKernelKey(result.decode, op, ropeKey);
    if (phaseResult.changed) {
      result.decode = phaseResult.steps;
      changed = true;
    }
  }

  const attentionStep = (result.decode || []).find((entry) => Array.isArray(entry) && entry[0] === 'attention');
  const attentionKernel = attentionStep ? result.kernels[attentionStep[1]] : null;
  const attentionEntry = deriveF16AttentionKernelEntry(attentionKernel);
  if (attentionStep && attentionEntry) {
    const attentionKey = deriveKernelKey(result.kernels, attentionStep[1], '_gemma4_f16');
    result.kernels[attentionKey] = attentionEntry;
    const phaseResult = replacePhaseStepKernelKey(result.decode, 'attention', attentionKey);
    if (phaseResult.changed) {
      result.decode = phaseResult.steps;
      changed = true;
    }
  }

  const oProjStep = (result.decode || []).find((entry) => Array.isArray(entry) && entry[0] === 'o_proj');
  const oProjKernel = oProjStep ? result.kernels[oProjStep[1]] : null;
  if (oProjStep && oProjKernel) {
    const oProjKey = deriveKernelKey(result.kernels, oProjStep[1], '_gemma4_f32_boundary');
    result.kernels[oProjKey] = deriveKernelEntryWithPrecision(oProjKernel, {
      inputDtype: 'f32',
      outputDtype: 'f32',
    });
    const phaseResult = replacePhaseStepKernelKey(result.decode, 'o_proj', oProjKey);
    if (phaseResult.changed) {
      result.decode = phaseResult.steps;
      changed = true;
    }
  }

  return changed ? result : null;
}

export const GEMMA4_12B_PREFILL_F32_PROJECTION_OPS = new Set([
  'q_proj',
  'k_proj',
  'v_proj',
  'gate_proj',
  'up_proj',
]);

export function remapGemma412BPrefillProjectionEntries(entries, f16Key, f32Key) {
  if (!Array.isArray(entries) || !f16Key || !f32Key) {
    return { entries, changed: false };
  }

  let changed = false;
  const remapStep = (step) => {
    if (!Array.isArray(step) || !LAYER_PROJECTION_OPS.has(step[0])) {
      return step;
    }
    const targetKey = GEMMA4_12B_PREFILL_F32_PROJECTION_OPS.has(step[0])
      ? f32Key
      : f16Key;
    if (step[1] === targetKey) {
      return step;
    }
    const replacement = [...step];
    replacement[1] = targetKey;
    changed = true;
    return replacement;
  };

  const nextEntries = entries.map((entry) => {
    if (Array.isArray(entry)) {
      return remapStep(entry);
    }
    if (!entry || typeof entry !== 'object' || !Array.isArray(entry.steps)) {
      return entry;
    }
    return {
      ...entry,
      steps: entry.steps.map((step) => remapStep(step)),
    };
  });

  return { entries: nextEntries, changed };
}

export function remapGemma412BStableBoundaryEntries(result, sourceGraph) {
  let changed = false;

  const replaceStepWithSourceEntry = (phaseName, op, precision) => {
    const phase = result[phaseName];
    const sourcePhase = sourceGraph[phaseName];
    const step = findPhaseStep(phase, op);
    const sourceStep = findPhaseStep(sourcePhase, op);
    const sourceKey = sourceStep?.[1] ?? step?.[1] ?? null;
    const sourceEntry = sourceKey ? sourceGraph.kernels[sourceKey] : null;
    if (!step || !sourceKey || !sourceEntry) {
      return;
    }
    const stableKey = deriveKernelKey(result.kernels, sourceKey, '_gemma4_12b_stable');
    result.kernels[stableKey] = precision
      ? deriveKernelEntryWithPrecision(sourceEntry, precision)
      : { ...sourceEntry };
    const phaseResult = replacePhaseStepKernelKey(phase, op, stableKey);
    if (phaseResult.changed) {
      result[phaseName] = phaseResult.steps;
      changed = true;
    }
  };

  for (const op of ['q_proj', 'k_proj', 'v_proj', 'rope_q', 'rope_k']) {
    replaceStepWithSourceEntry('decode', op, { inputDtype: 'f32', outputDtype: 'f32' });
  }
  replaceStepWithSourceEntry('decode', 'attention', {
    activationDtype: 'f32',
    kvDtype: 'f16',
    outputDtype: 'f32',
  });
  for (const op of ['final_norm', 'lm_head', 'lm_head_prefill', 'sample']) {
    replaceStepWithSourceEntry('postLayer', op, null);
  }

  return changed;
}

export function useGemma4TextF16ActivationsForLane(graph, ctx, options) {
  const narrowed = narrowToF16Activations(graph, ctx);
  const result = narrowed ?? cloneGraph(graph);
  let changed = narrowed != null;

  const replaceKernelEntry = (key, entry) => {
    if (!key || !entry) {
      return;
    }
    result.kernels[key] = entry;
    changed = true;
  };
  const stableTextBoundary = options?.stableTextBoundary === true;

  const derivePrefillAttentionEntry = (entry) => {
    if (
      stableTextBoundary
      && typeof entry?.kernel === 'string'
      && entry.kernel === 'attention_small_f16.wgsl'
    ) {
      return deriveKernelEntryWithPrecision(
        deriveKernelEntry(entry, 'attention_head256_f16kv.wgsl', 'main'),
        { activationDtype: 'f32', kvDtype: 'f16', outputDtype: 'f32' }
      );
    }
    if (
      stableTextBoundary
      && typeof entry?.kernel === 'string'
      && entry.kernel.endsWith('_f16kv.wgsl')
    ) {
      return deriveKernelEntryWithPrecision(
        entry,
        { activationDtype: 'f32', kvDtype: 'f16', outputDtype: 'f32' }
      );
    }
    return deriveF16AttentionKernelEntry(entry);
  };

  const embedStep = findPhaseStep(result.preLayer, 'embed');
  const embedKey = embedStep?.[1] ?? null;
  const embedEntry = embedKey ? result.kernels[embedKey] : null;
  if (embedEntry?.kernel === 'gather_f16.wgsl') {
    replaceKernelEntry(
      embedKey,
      deriveKernelEntryWithPrecision(
        deriveKernelEntry(embedEntry, 'gather_f16_vec4_f16_out.wgsl', 'gather_vec4_f16_out'),
        { inputDtype: 'f16', outputDtype: 'f16' }
      )
    );
  }

  const decodeProjectionStep = findPhaseStep(result.decode, 'q_proj');
  const decodeProjectionKey = decodeProjectionStep?.[1] ?? null;
  replaceKernelEntry(
    decodeProjectionKey,
    deriveQ4DecodeF16KernelEntry(result.kernels[decodeProjectionKey])
  );

  const prefillProjectionStep = findPhaseStep(result.prefill, 'q_proj');
  const prefillProjectionKey = prefillProjectionStep?.[1] ?? null;
  const sourcePrefillProjectionEntry = result.kernels[prefillProjectionKey];
  const prefillProjectionF16Entry = stableTextBoundary
    ? (
        deriveQ4WideTilePrefillF16KernelEntry(sourcePrefillProjectionEntry)
        ?? deriveQ4PrefillF16AccumKernelEntry(sourcePrefillProjectionEntry)
        ?? deriveQ4PrefillF16KernelEntry(sourcePrefillProjectionEntry)
      )
    : (
        deriveQ4PrefillF16AccumKernelEntry(sourcePrefillProjectionEntry)
        ?? deriveQ4WideTilePrefillF16KernelEntry(sourcePrefillProjectionEntry)
        ?? deriveQ4PrefillF16KernelEntry(sourcePrefillProjectionEntry)
      );
  if (stableTextBoundary && prefillProjectionKey && prefillProjectionF16Entry) {
    const prefillProjectionF16Key = deriveKernelKey(result.kernels, prefillProjectionKey, '_gemma4_f16');
    result.kernels[prefillProjectionF16Key] = prefillProjectionF16Entry;
    changed = true;
    const remapped = remapGemma412BPrefillProjectionEntries(
      result.prefill,
      prefillProjectionF16Key,
      prefillProjectionKey
    );
    result.prefill = remapped.entries;
    changed = changed || remapped.changed;
  } else {
    replaceKernelEntry(
      prefillProjectionKey,
      prefillProjectionF16Entry
    );
  }

  const replacePrefillAttentionEntries = (entries) => {
    for (const entry of entries || []) {
      if (Array.isArray(entry)) {
        if (entry[0] !== 'attention') {
          continue;
        }
        replaceKernelEntry(
          entry[1],
          derivePrefillAttentionEntry(result.kernels[entry[1]])
        );
        continue;
      }
      if (entry && typeof entry === 'object' && Array.isArray(entry.steps)) {
        replacePrefillAttentionEntries(entry.steps);
      }
    }
  };
  replacePrefillAttentionEntries(result.prefill);

  const finalNormStep = findPhaseStep(result.postLayer, 'final_norm');
  const finalNormKey = finalNormStep?.[1] ?? null;
  const finalNormEntry = finalNormKey ? result.kernels[finalNormKey] : null;
  if (finalNormEntry?.kernel === 'rmsnorm.wgsl') {
    replaceKernelEntry(
      finalNormKey,
      deriveKernelEntryWithPrecision(
        deriveKernelEntry(finalNormEntry, 'rmsnorm_f16.wgsl', finalNormEntry.entry),
        { inputDtype: 'f16', outputDtype: 'f16' }
      )
    );
  }

  const lmHeadStep = findPhaseStep(result.postLayer, 'lm_head');
  const lmHeadKey = lmHeadStep?.[1] ?? null;
  replaceKernelEntry(
    lmHeadKey,
    deriveLmHeadDecodeF16KernelEntry(result.kernels[lmHeadKey])
  );

  const lmHeadPrefillStep = findPhaseStep(result.postLayer, 'lm_head_prefill');
  const lmHeadPrefillKey = lmHeadPrefillStep?.[1] ?? null;
  const lmHeadPrefillEntry = lmHeadPrefillKey ? result.kernels[lmHeadPrefillKey] : null;
  if (
    lmHeadPrefillEntry?.kernel === 'matmul_f16w_f32a.wgsl'
    || lmHeadPrefillEntry?.kernel === 'matmul_f16w_f32a_tiled.wgsl'
  ) {
    replaceKernelEntry(
      lmHeadPrefillKey,
      deriveKernelEntryWithPrecision(
        deriveKernelEntry(lmHeadPrefillEntry, 'matmul_f16_tiled.wgsl', 'main'),
        { inputDtype: 'f16', outputDtype: 'f16' }
      )
    );
  }

  if (stableTextBoundary) {
    changed = remapGemma412BStableBoundaryEntries(result, graph) || changed;
  }

  return changed ? result : null;
}

export function useGemma4Int4PleAf16Activations(graph, ctx) {
  const narrowed = narrowToF16Activations(graph, ctx);
  const result = narrowed ?? cloneGraph(graph);
  let changed = narrowed != null;

  const replaceKernelEntry = (key, entry) => {
    if (!key || !entry) {
      return;
    }
    result.kernels[key] = entry;
    changed = true;
  };

  const embedStep = findPhaseStep(result.preLayer, 'embed');
  const embedKey = embedStep?.[1] ?? null;
  const embedEntry = embedKey ? result.kernels[embedKey] : null;
  if (embedEntry?.kernel === 'gather_f16.wgsl') {
    replaceKernelEntry(
      embedKey,
      deriveKernelEntryWithPrecision(
        deriveKernelEntry(embedEntry, 'gather_f16_vec4_f16_out.wgsl', 'gather_vec4_f16_out'),
        { inputDtype: 'f16', outputDtype: 'f16' }
      )
    );
  }

  const decodeProjectionStep = findPhaseStep(result.decode, 'q_proj');
  const decodeProjectionKey = decodeProjectionStep?.[1] ?? null;
  replaceKernelEntry(
    decodeProjectionKey,
    deriveQ4DecodeF16KernelEntry(result.kernels[decodeProjectionKey])
  );

  const prefillProjectionStep = findPhaseStep(result.prefill, 'q_proj');
  const prefillProjectionKey = prefillProjectionStep?.[1] ?? null;
  replaceKernelEntry(
    prefillProjectionKey,
    deriveQ4WideTilePrefillF16KernelEntry(result.kernels[prefillProjectionKey])
      ?? deriveQ4PrefillF16AccumKernelEntry(result.kernels[prefillProjectionKey])
      ?? deriveQ4PrefillF16KernelEntry(result.kernels[prefillProjectionKey])
  );

  const replacePrefillAttentionEntries = (entries) => {
    for (const entry of entries || []) {
      if (Array.isArray(entry)) {
        if (entry[0] !== 'attention') {
          continue;
        }
        replaceKernelEntry(
          entry[1],
          deriveF16AttentionKernelEntry(result.kernels[entry[1]])
        );
        continue;
      }
      if (entry && typeof entry === 'object' && Array.isArray(entry.steps)) {
        replacePrefillAttentionEntries(entry.steps);
      }
    }
  };
  replacePrefillAttentionEntries(result.prefill);

  const finalNormStep = findPhaseStep(result.postLayer, 'final_norm');
  const finalNormKey = finalNormStep?.[1] ?? null;
  const finalNormEntry = finalNormKey ? result.kernels[finalNormKey] : null;
  if (finalNormEntry?.kernel === 'rmsnorm.wgsl') {
    replaceKernelEntry(
      finalNormKey,
      deriveKernelEntryWithPrecision(
        deriveKernelEntry(finalNormEntry, 'rmsnorm_f16.wgsl', finalNormEntry.entry),
        { inputDtype: 'f16', outputDtype: 'f16' }
      )
    );
  }

  const lmHeadStep = findPhaseStep(result.postLayer, 'lm_head');
  const lmHeadKey = lmHeadStep?.[1] ?? null;
  replaceKernelEntry(
    lmHeadKey,
    deriveLmHeadDecodeF16KernelEntry(result.kernels[lmHeadKey])
  );

  const lmHeadPrefillStep = findPhaseStep(result.postLayer, 'lm_head_prefill');
  const lmHeadPrefillKey = lmHeadPrefillStep?.[1] ?? null;
  const lmHeadPrefillEntry = lmHeadPrefillKey ? result.kernels[lmHeadPrefillKey] : null;
  if (
    lmHeadPrefillEntry?.kernel === 'matmul_f16w_f32a.wgsl'
    || lmHeadPrefillEntry?.kernel === 'matmul_f16w_f32a_tiled.wgsl'
  ) {
    replaceKernelEntry(
      lmHeadPrefillKey,
      deriveKernelEntryWithPrecision(
        deriveKernelEntry(lmHeadPrefillEntry, 'matmul_f16_tiled.wgsl', 'main'),
        { inputDtype: 'f16', outputDtype: 'f16' }
      )
    );
  }

  return changed ? result : null;
}
