import { isGemma4PerLayerEmbedTensor } from "../tensor-transform.js";
import { getNestedTextConfig } from "../artifact-identity.js";
import { modelHasMoETensors } from "../manifest-builder.js";
import { normalizeTensorName } from "../tokenizer-contract.js";

export function validateInt4PleMaterializationContract(tensorLocations, inference, modelId) {
  const materialization = inference?.session?.perLayerInputs?.materialization;
  if (materialization !== 'gpu_split_tables') {
    return;
  }
  for (const [name, location] of Object.entries(tensorLocations ?? {})) {
    const sourceTransform = location?.sourceTransform ?? null;
    if (
      isGemma4PerLayerEmbedTensor(name)
      && sourceTransform?.kind === 'litert_axis_dequant'
      && String(sourceTransform?.sourceDtype ?? '').toUpperCase() === 'INT4'
    ) {
      throw new Error(
        `Manifest "${modelId}" Gemma 4 INT4 PLE tensor "${name}" cannot use ` +
        'inference.session.perLayerInputs.materialization="gpu_split_tables". ' +
        'Use materialization="range_backed" or disable INT4 PLE quantization.'
      );
    }
  }
}

function resolveGemma4TextConfig(rawConfig) {
  const textConfig = getNestedTextConfig(rawConfig);
  const modelType = String(textConfig?.model_type ?? rawConfig?.model_type ?? '').trim().toLowerCase();
  if (
    modelType !== 'gemma4'
    && modelType !== 'gemma4_text'
    && modelType !== 'gemma4_unified'
    && modelType !== 'gemma4_unified_text'
  ) {
    return null;
  }
  return textConfig ?? rawConfig ?? null;
}

function collectGemma4UnsupportedTensorFlags(tensors) {
  const names = Array.isArray(tensors) ? tensors.map((tensor) => normalizeTensorName(tensor).toLowerCase()) : [];
  const flags = [];
  if (names.some((name) => name.includes('.experts.gate_up_proj'))) {
    flags.push('experts.gate_up_proj');
  }
  if (names.some((name) => name.includes('.experts.down_proj'))) {
    flags.push('experts.down_proj');
  }
  if (names.some((name) => name.includes('.router.per_expert_scale'))) {
    flags.push('router.per_expert_scale');
  }
  if (names.some((name) => name.includes('.router.scale'))) {
    flags.push('router.scale');
  }
  if (names.some((name) => name.includes('.post_feedforward_layernorm_1.'))) {
    flags.push('post_feedforward_layernorm_1');
  }
  if (names.some((name) => name.includes('.post_feedforward_layernorm_2.'))) {
    flags.push('post_feedforward_layernorm_2');
  }
  if (names.some((name) => name.includes('.pre_feedforward_layernorm_2.'))) {
    flags.push('pre_feedforward_layernorm_2');
  }
  return flags;
}

export function assertSupportedGemma4Conversion(model, tensors, modelId) {
  const rawConfig = model?.config ?? null;
  const textConfig = resolveGemma4TextConfig(rawConfig);
  if (!textConfig) return;

  const hiddenSizePerLayerInput = Number(textConfig.hidden_size_per_layer_input ?? 0);
  if (Number.isFinite(hiddenSizePerLayerInput) && hiddenSizePerLayerInput > 0) {
    const names = Array.isArray(tensors)
      ? tensors.map((tensor) => String(tensor?.name ?? '').trim())
      : [];
    const requiredNames = [
      'embed_tokens_per_layer.weight',
      'per_layer_input_gate.weight',
      'per_layer_projection.weight',
      'post_per_layer_input_norm.weight',
      'per_layer_model_projection.weight',
      'per_layer_projection_norm.weight',
    ];
    const missing = requiredNames.filter((suffix) => !names.some((name) => name.endsWith(suffix)));
    if (missing.length > 0) {
      throw new Error(
        `Gemma 4 model "${modelId}" declares hidden_size_per_layer_input=${hiddenSizePerLayerInput}, ` +
        `but the checkpoint is missing required per-layer input tensors: ${missing.join(', ')}.`
      );
    }
  }

  if (textConfig.enable_moe_block !== true) {
    return;
  }

  const unsupportedFlags = collectGemma4UnsupportedTensorFlags(tensors);
  if (unsupportedFlags.length === 0 && !modelHasMoETensors({ tensors })) {
    return;
  }

  throw new Error(
    `Gemma 4 model "${modelId}" is not supported yet: Gemma 4 MoE decoder blocks require ` +
    'Gemma-specific router scaling and dual dense+MoE FFN execution, but current Doppler MoE runtime ' +
    'only supports Mixtral/GPT-OSS semantics. ' +
    `Detected: ${unsupportedFlags.length > 0 ? unsupportedFlags.join(', ') : 'Gemma 4 MoE tensors'}.`
  );
}
