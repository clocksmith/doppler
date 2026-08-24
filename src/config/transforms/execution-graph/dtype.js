

export function cloneGraph(graph) {
  return structuredClone(graph);
}

export const KERNEL_FILE_PRECISION_PATCHES = new Map([
  ['matmul_gemv_subgroup_f16a.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['matmul_f16.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['matmul_f16_tiled.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['silu_f16.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['fused_matmul_q4_multicol_f16.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['fused_matmul_q4_multicol_f16a.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['fused_matmul_q4_batched_f16acc_f16a.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['fused_matmul_q4.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['fused_matmul_q4_batched.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['fused_matmul_q4_batched_multicol_shared.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['fused_matmul_q4_widetile_f16a.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['matmul_gemv_subgroup.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['matmul_f16w_f32a.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['matmul_f16w_f32a_tiled.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['matmul_f32.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['gather_f16_f16_out.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['gather_f16_vec4_f16_out.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
]);

export function deriveKernelEntry(base, newFile, newEntry, constants) {
  const derived = { ...base, kernel: newFile, entry: newEntry, digest: null };
  if (constants === null) {
    delete derived.constants;
  } else if (constants !== undefined) {
    derived.constants = { ...constants };
  }
  const precision = deriveKernelPrecision(base, newFile);
  if (precision) {
    derived.precision = precision;
  } else {
    delete derived.precision;
  }
  return derived;
}

export function deriveKernelPrecision(base, newFile) {
  const precision = base.precision ? { ...base.precision } : {};
  const precisionPatch = KERNEL_FILE_PRECISION_PATCHES.get(newFile);
  if (precisionPatch) {
    Object.assign(precision, precisionPatch);
  }
  if (!String(newFile).startsWith('attention')) {
    return Object.keys(precision).length > 0 ? precision : null;
  }
  if (newFile.includes('_f16kv')) {
    precision.activationDtype = 'f32';
    precision.kvDtype = 'f16';
    return precision;
  }
  if (newFile.includes('_f16')) {
    precision.activationDtype = 'f16';
    precision.kvDtype = 'f16';
    return precision;
  }
  precision.activationDtype = 'f32';
  precision.kvDtype = 'f32';
  return precision;
}

export function deriveKernelKey(kernels, baseKey, suffix) {
  const candidate = `${baseKey}${suffix}`;
  if (!kernels[candidate]) {
    return candidate;
  }
  let counter = 2;
  while (kernels[`${candidate}_${counter}`]) {
    counter++;
  }
  return `${candidate}_${counter}`;
}

export function replacePhaseStepKernelKey(steps, op, replacementKernelKey) {
  if (!Array.isArray(steps) || steps.length === 0 || !replacementKernelKey) {
    return { steps, changed: false };
  }
  let changed = false;
  const nextSteps = steps.map((step) => {
    if (!Array.isArray(step) || step[0] !== op) {
      return step;
    }
    if (step[1] === replacementKernelKey) {
      return step;
    }
    const replacement = [...step];
    replacement[1] = replacementKernelKey;
    changed = true;
    return replacement;
  });
  return { steps: nextSteps, changed };
}

export function findPhaseStep(steps, op) {
  if (!Array.isArray(steps) || !op) {
    return null;
  }
  for (const entry of steps) {
    if (Array.isArray(entry)) {
      if (entry[0] === op) {
        return entry;
      }
      continue;
    }
    if (!entry || typeof entry !== 'object' || !Array.isArray(entry.steps)) {
      continue;
    }
    const nested = findPhaseStep(entry.steps, op);
    if (nested) {
      return nested;
    }
  }
  return null;
}

export function deriveKernelEntryWithPrecision(base, precision) {
  return {
    ...base,
    precision: {
      ...(base.precision ?? {}),
      ...precision,
    },
  };
}

export const ATTENTION_F16KV_TO_F16_MAP = new Map([
  ['attention_decode_online_f16kv.wgsl', 'attention_decode_online_f16.wgsl'],
  ['attention_decode_chunked_f16kv.wgsl', 'attention_decode_chunked_f16.wgsl'],
  ['attention_small_f16kv.wgsl', 'attention_small_f16.wgsl'],
  ['attention_streaming_f16kv.wgsl', 'attention_streaming_f16.wgsl'],
  ['attention_head512_f16kv.wgsl', 'attention_head512_f16.wgsl'],
]);

export function deriveF16AttentionKernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const replacement = ATTENTION_F16KV_TO_F16_MAP.get(base.kernel);
  if (!replacement) {
    return null;
  }
  return deriveKernelEntryWithPrecision(
    deriveKernelEntry(base, replacement, base.entry),
    { activationDtype: 'f16', kvDtype: 'f16' }
  );
}

export function deriveLinearDecodeF16KernelEntry(base) {
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (base.kernel === 'fused_matmul_q4.wgsl' && base.entry === 'main_multicol') {
    return {
      ...deriveKernelEntry(base, 'fused_matmul_q4_multicol_f16a.wgsl', 'main_multicol_f16a'),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  if (
    (base.kernel === 'fused_matmul_q4_multicol_f16.wgsl' && base.entry === 'main_multicol_f16')
    || (base.kernel === 'fused_matmul_q4_multicol_f16a.wgsl' && base.entry === 'main_multicol_f16a')
  ) {
    return deriveKernelEntryWithPrecision(base, precision);
  }
  return null;
}

export function deriveLmHeadDecodeF16KernelEntry(base) {
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (base.kernel === 'matmul_gemv_subgroup.wgsl' && base.entry === 'main_multicol') {
    return {
      ...deriveKernelEntry(base, 'matmul_gemv_subgroup_f16a.wgsl', 'main_multicol'),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  if (base.kernel === 'matmul_gemv_subgroup_f16a.wgsl' && base.entry === 'main_multicol') {
    return deriveKernelEntryWithPrecision(base, precision);
  }
  return null;
}

export function deriveQ4DecodeF16KernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (base.kernel === 'fused_matmul_q4.wgsl') {
    return {
      ...deriveKernelEntry(base, 'fused_matmul_q4_multicol_f16a.wgsl', 'main_multicol_f16a', null),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  if (
    base.kernel === 'fused_matmul_q4_multicol_f16.wgsl'
    || base.kernel === 'fused_matmul_q4_multicol_f16a.wgsl'
  ) {
    return deriveKernelEntryWithPrecision(base, precision);
  }
  return null;
}

export function deriveQ4PrefillF16KernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (base.kernel === 'fused_matmul_q4_widetile_f16a.wgsl') {
    return deriveKernelEntryWithPrecision(base, precision);
  }
  if (base.kernel.startsWith('fused_matmul_q4_batched')) {
    return {
      ...deriveKernelEntry(base, 'fused_matmul_q4_batched_f16a.wgsl', 'main_batched_f16a', null),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  return null;
}

export function deriveQ4WideTilePrefillF16KernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (base.kernel === 'fused_matmul_q4_widetile.wgsl') {
    return {
      ...deriveKernelEntry(base, 'fused_matmul_q4_widetile_f16a.wgsl', 'main', null),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  if (base.kernel === 'fused_matmul_q4_widetile_f16a.wgsl') {
    return deriveKernelEntryWithPrecision(base, precision);
  }
  return null;
}

export function deriveQ4PrefillF16AccumKernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (
    base.kernel === 'fused_matmul_q4_widetile.wgsl'
    || base.kernel === 'fused_matmul_q4_widetile_f16a.wgsl'
    || base.kernel.startsWith('fused_matmul_q4_batched')
  ) {
    return {
      ...deriveKernelEntry(base, 'fused_matmul_q4_batched_f16acc_f16a.wgsl', 'main_batched_f16acc_f16a', null),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  return null;
}

export const F16_TO_F32_ACTIVATION_MAP = new Map([
  ['rmsnorm_f16.wgsl', 'rmsnorm.wgsl'],
  ['rope_f16.wgsl', 'rope.wgsl'],
  ['residual_f16.wgsl', 'residual.wgsl'],
  ['gelu_f16.wgsl', 'gelu.wgsl'],
  ['silu_f16.wgsl', 'silu.wgsl'],
  ['sample_f16.wgsl', 'sample.wgsl'],
  ['gather_f16.wgsl', 'gather.wgsl'],
  ['gather_f16_f16_out.wgsl', 'gather.wgsl'],
  ['gather_f16_vec4_f16_out.wgsl', 'gather.wgsl'],
  ['matmul_gemv_subgroup_f16a.wgsl', 'matmul_gemv_subgroup.wgsl'],
  ['matmul_f16.wgsl', 'matmul_f16w_f32a.wgsl'],
  ['matmul_f16_tiled.wgsl', 'matmul_f16w_f32a_tiled.wgsl'],
  ['attention_decode_online_f16.wgsl', 'attention_decode_online_f16kv.wgsl'],
  ['attention_decode_chunked_f16.wgsl', 'attention_decode_chunked_f16kv.wgsl'],
  ['attention_small_f16.wgsl', 'attention_small_f16kv.wgsl'],
  ['attention_streaming_f16.wgsl', 'attention_streaming_f16kv.wgsl'],
  ['attention_head512_f16.wgsl', 'attention_head512_f16kv.wgsl'],
]);

export const F32_TO_F16_ACTIVATION_MAP = new Map(
  Array.from(F16_TO_F32_ACTIVATION_MAP.entries(), ([from, to]) => [to, from])
);

export function hasExplicitF32ActivationContract(entry) {
  const precision = entry?.precision;
  if (!precision || typeof precision !== 'object') {
    return false;
  }
  return precision.activationDtype === 'f32'
    || precision.inputDtype === 'f32'
    || precision.outputDtype === 'f32';
}

export function narrowToF16Activations(graph, ctx) {
  if (ctx.activationDtype !== 'f16' || ctx.capabilities?.hasF16 !== true) {
    return null;
  }

  const hasTargetShader = Object.values(graph.kernels).some(
    (entry) => !hasExplicitF32ActivationContract(entry) && F32_TO_F16_ACTIVATION_MAP.has(entry.kernel)
  );
  if (!hasTargetShader) {
    return null;
  }

  const result = cloneGraph(graph);
  for (const [key, entry] of Object.entries(result.kernels)) {
    if (hasExplicitF32ActivationContract(entry)) {
      continue;
    }
    const replacement = F32_TO_F16_ACTIVATION_MAP.get(entry.kernel);
    if (replacement !== undefined) {
      result.kernels[key] = deriveKernelEntry(entry, replacement, entry.entry);
    }
  }
  return result;
}

export function useQwenDecodeF16Matmuls(graph, ctx) {
  const result = cloneGraph(graph);
  let changed = false;

  for (const op of ['gate_proj', 'up_proj']) {
    const stepIndex = (result.decode || []).findIndex((entry) => Array.isArray(entry) && entry[0] === op);
    if (stepIndex === -1) {
      continue;
    }
    const step = result.decode[stepIndex];
    const kernelKey = step[1];
    const kernelEntry = result.kernels[kernelKey];
    if (!kernelEntry) {
      continue;
    }
    const derivedEntry = deriveLinearDecodeF16KernelEntry(kernelEntry);
    if (!derivedEntry) {
      continue;
    }
    const derivedKey = deriveKernelKey(result.kernels, kernelKey, '_decode_f16out');
    result.kernels[derivedKey] = derivedEntry;
    const replacement = [...step];
    replacement[1] = derivedKey;
    result.decode = [
      ...result.decode.slice(0, stepIndex),
      replacement,
      ...result.decode.slice(stepIndex + 1),
    ];
    changed = true;
  }

  const postLayerResult = replacePhaseStepKernelKey(
    result.postLayer ?? [],
    'lm_head',
    (() => {
      const lmHeadStep = (result.postLayer || []).find((entry) => Array.isArray(entry) && entry[0] === 'lm_head');
      if (!lmHeadStep) {
        return null;
      }
      const lmHeadKernelKey = lmHeadStep[1];
      const lmHeadKernel = result.kernels[lmHeadKernelKey];
      if (!lmHeadKernel) {
        return null;
      }
      const derivedEntry = deriveLmHeadDecodeF16KernelEntry(lmHeadKernel);
      if (!derivedEntry) {
        return null;
      }
      const derivedKey = deriveKernelKey(result.kernels, lmHeadKernelKey, '_decode_f16out');
      result.kernels[derivedKey] = derivedEntry;
      return derivedKey;
    })()
  );
  if (postLayerResult.changed) {
    result.postLayer = postLayerResult.steps;
    changed = true;
  }

  return changed ? result : null;
}

export function useQwenF16PrimaryMatmuls(graph, ctx) {
  const layerTypes = Array.isArray(ctx.layerTypes) ? ctx.layerTypes : null;
  if (!layerTypes || layerTypes.length === 0) {
    return null;
  }

  const result = cloneGraph(graph);
  let changed = false;

  for (const [phaseName, op] of [['decode', 'attention'], ['prefill', 'attention']]) {
    const phaseSteps = result[phaseName] || [];
    const step = phaseSteps.find((entry) => Array.isArray(entry) && entry[0] === op);
    const kernelKey = step?.[1];
    const kernelEntry = kernelKey ? result.kernels[kernelKey] : null;
    const derivedEntry = deriveF16AttentionKernelEntry(kernelEntry);
    if (!step || !kernelKey || !derivedEntry) {
      continue;
    }
    const derivedKey = deriveKernelKey(result.kernels, kernelKey, '_primary_f16');
    result.kernels[derivedKey] = derivedEntry;
    const phaseResult = replacePhaseStepKernelKey(phaseSteps, op, derivedKey);
    if (phaseResult.changed) {
      result[phaseName] = phaseResult.steps;
      changed = true;
    }
  }

  const decodeProjectionStep = (result.decode || []).find((entry) => Array.isArray(entry) && entry[0] === 'q_proj');
  const decodeProjectionKernel = decodeProjectionStep ? result.kernels[decodeProjectionStep[1]] : null;
  const decodeProjectionEntry = deriveQ4DecodeF16KernelEntry(decodeProjectionKernel);
  if (decodeProjectionStep && decodeProjectionEntry) {
    const decodeProjectionKey = deriveKernelKey(result.kernels, decodeProjectionStep[1], '_primary_f16');
    result.kernels[decodeProjectionKey] = decodeProjectionEntry;
    for (const op of ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']) {
      const phaseResult = replacePhaseStepKernelKey(result.decode, op, decodeProjectionKey);
      if (phaseResult.changed) {
        result.decode = phaseResult.steps;
        changed = true;
      }
    }
  }

  const prefillProjectionStep = (result.prefill || []).find((entry) => Array.isArray(entry) && entry[0] === 'q_proj');
  const prefillProjectionKernel = prefillProjectionStep ? result.kernels[prefillProjectionStep[1]] : null;
  const prefillProjectionEntry = deriveQ4PrefillF16KernelEntry(prefillProjectionKernel);
  if (prefillProjectionStep && prefillProjectionEntry) {
    const prefillProjectionKey = deriveKernelKey(result.kernels, prefillProjectionStep[1], '_primary_f16');
    result.kernels[prefillProjectionKey] = prefillProjectionEntry;
    for (const op of ['q_proj', 'k_proj', 'v_proj']) {
      const phaseResult = replacePhaseStepKernelKey(result.prefill, op, prefillProjectionKey);
      if (phaseResult.changed) {
        result.prefill = phaseResult.steps;
        changed = true;
      }
    }
  }

  for (const [phaseName, op] of [['decode', 'o_proj'], ['prefill', 'o_proj']]) {
    const phaseSteps = result[phaseName] || [];
    const step = phaseSteps.find((entry) => Array.isArray(entry) && entry[0] === op);
    const kernelKey = step?.[1];
    const kernelEntry = kernelKey ? result.kernels[kernelKey] : null;
    if (!step || !kernelKey || !kernelEntry) {
      continue;
    }
    const boundaryKey = deriveKernelKey(result.kernels, kernelKey, '_primary_f32_boundary');
    result.kernels[boundaryKey] = deriveKernelEntryWithPrecision(kernelEntry, {
      inputDtype: 'f32',
      outputDtype: 'f32',
    });
    const phaseResult = replacePhaseStepKernelKey(phaseSteps, op, boundaryKey);
    if (phaseResult.changed) {
      result[phaseName] = phaseResult.steps;
      changed = true;
    }
  }

  const lmHeadStep = (result.postLayer || []).find((entry) => Array.isArray(entry) && entry[0] === 'lm_head');
  const lmHeadKernel = lmHeadStep ? result.kernels[lmHeadStep[1]] : null;
  const lmHeadEntry = deriveLmHeadDecodeF16KernelEntry(lmHeadKernel);
  if (lmHeadStep && lmHeadEntry) {
    const lmHeadKey = deriveKernelKey(result.kernels, lmHeadStep[1], '_primary_f16');
    result.kernels[lmHeadKey] = lmHeadEntry;
    const phaseResult = replacePhaseStepKernelKey(result.postLayer, 'lm_head', lmHeadKey);
    if (phaseResult.changed) {
      result.postLayer = phaseResult.steps;
      changed = true;
    }
  }

  return changed ? result : null;
}

export function useQwen36F16Activations(graph, ctx) {
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
  const replaceOps = (phaseName, ops, kernelKey) => {
    for (const op of ops) {
      const phaseResult = replacePhaseStepKernelKey(result[phaseName], op, kernelKey);
      if (phaseResult.changed) {
        result[phaseName] = phaseResult.steps;
        changed = true;
      }
    }
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
  const decodeProjectionEntry = deriveQ4DecodeF16KernelEntry(result.kernels[decodeProjectionKey]);
  if (decodeProjectionEntry) {
    const f16Key = deriveKernelKey(result.kernels, decodeProjectionKey, '_qwen36_f16');
    result.kernels[f16Key] = decodeProjectionEntry;
    replaceOps('decode', ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], f16Key);
  }

  const prefillProjectionStep = findPhaseStep(result.prefill, 'q_proj');
  const prefillProjectionKey = prefillProjectionStep?.[1] ?? null;
  const prefillProjectionEntry =
    deriveQ4WideTilePrefillF16KernelEntry(result.kernels[prefillProjectionKey])
    ?? deriveQ4PrefillF16AccumKernelEntry(result.kernels[prefillProjectionKey])
    ?? deriveQ4PrefillF16KernelEntry(result.kernels[prefillProjectionKey]);
  if (prefillProjectionEntry) {
    const f16Key = deriveKernelKey(result.kernels, prefillProjectionKey, '_qwen36_f16');
    result.kernels[f16Key] = prefillProjectionEntry;
    replaceOps('prefill', ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], f16Key);
  }

  const lmHeadStep = findPhaseStep(result.postLayer, 'lm_head');
  const lmHeadKey = lmHeadStep?.[1] ?? null;
  replaceKernelEntry(
    lmHeadKey,
    deriveQ4DecodeF16KernelEntry(result.kernels[lmHeadKey])
  );

  const lmHeadPrefillStep = findPhaseStep(result.postLayer, 'lm_head_prefill');
  const lmHeadPrefillKey = lmHeadPrefillStep?.[1] ?? null;
  const lmHeadPrefillEntry = lmHeadPrefillKey ? result.kernels[lmHeadPrefillKey] : null;
  const lmHeadPrefillF16Entry =
    deriveQ4WideTilePrefillF16KernelEntry(lmHeadPrefillEntry)
    ?? deriveQ4PrefillF16AccumKernelEntry(lmHeadPrefillEntry)
    ?? deriveQ4PrefillF16KernelEntry(lmHeadPrefillEntry);
  if (lmHeadPrefillF16Entry) {
    replaceKernelEntry(lmHeadPrefillKey, lmHeadPrefillF16Entry);
  } else if (
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
