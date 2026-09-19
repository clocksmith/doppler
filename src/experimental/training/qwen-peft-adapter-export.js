import { readBuffer } from '../../memory/buffer-pool.js';
import { serializeLoRASafetensors } from './export.js';
import {
  QWEN_PEFT_ATTENTION_PROJECTIONS,
  normalizeQwenPeftLayerTypes,
  normalizeQwenPeftTargetModules,
  qwenPeftExpectedProjections,
  transposeQwenPeftMatrix,
} from './qwen-peft-adapter-import.js';

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function finiteNumber(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be finite.`);
  }
  return parsed;
}

function sameShape(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Number(value) === Number(right[index]));
}

function targetPair(layer, projection) {
  const key = projection.replace(/_proj$/, '');
  if (QWEN_PEFT_ATTENTION_PROJECTIONS.includes(projection)) {
    return layer.inputs?.attention?.lora?.[key];
  }
  return layer.inputs?.mlp?.lora?.[key];
}

function peftTensorName(layerIndex, layerType, projection, kind) {
  const branch = layerType === 'full_attention'
    && QWEN_PEFT_ATTENTION_PROJECTIONS.includes(projection)
    ? 'self_attn'
    : 'mlp';
  return `base_model.model.model.language_model.layers.${layerIndex}.${branch}.${projection}.lora_${kind}.weight`;
}

function tensorShape(tensor, label) {
  const shape = Array.isArray(tensor?.shape) ? tensor.shape.map(Number) : [];
  if (!tensor?.buffer || tensor.dtype !== 'f32' || shape.length !== 2
    || shape.some((value) => !Number.isInteger(value) || value < 1)) {
    throw new Error(`${label} requires a positive rank-two F32 tensor.`);
  }
  return shape;
}

async function readF32Tensor(tensor, shape, label) {
  if (!sameShape(tensor.shape, shape)) {
    throw new Error(`${label} requires an F32 tensor with shape [${shape}].`);
  }
  const elementCount = shape[0] * shape[1];
  return new Float32Array(await readBuffer(tensor.buffer, elementCount * 4));
}

export async function exportQwenPeftAdapterFromLayers(layers, options) {
  const rank = positiveInteger(options?.rank, 'Qwen PEFT export rank');
  const alpha = finiteNumber(options?.alpha, 'Qwen PEFT export alpha');
  const dropout = finiteNumber(options?.dropout, 'Qwen PEFT export dropout');
  if (alpha <= 0) throw new Error('Qwen PEFT export alpha must be positive.');
  if (dropout < 0 || dropout >= 1) {
    throw new Error('Qwen PEFT export dropout must be in [0, 1).');
  }
  const baseModel = String(options?.baseModel || '').trim();
  if (!baseModel) throw new Error('Qwen PEFT export baseModel is required.');
  const layerTypes = normalizeQwenPeftLayerTypes(options?.layerTypes);
  const targetModules = normalizeQwenPeftTargetModules(options?.targetModules);
  if (!Array.isArray(layers) || layers.length !== layerTypes.length) {
    throw new Error('Qwen PEFT export layer count does not match layerTypes.');
  }

  const tensors = [];
  for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
    const layerType = layerTypes[layerIndex];
    if (layers[layerIndex]?.type !== layerType) {
      throw new Error(`Qwen PEFT export layer ${layerIndex} type mismatch.`);
    }
    for (const projection of qwenPeftExpectedProjections(layerType, targetModules)) {
      const pair = targetPair(layers[layerIndex], projection);
      if (!pair?.A || !pair?.B) {
        throw new Error(`Qwen PEFT export is missing ${layerIndex}.${projection}.`);
      }
      if (Number(pair.rank) !== rank || Number(pair.alpha) !== alpha) {
        throw new Error(`Qwen PEFT export adapter metadata mismatch at ${layerIndex}.${projection}.`);
      }
      const aShape = tensorShape(pair.A, `${layerIndex}.${projection}.lora_A`);
      const bShape = tensorShape(pair.B, `${layerIndex}.${projection}.lora_B`);
      if (aShape[1] !== rank || bShape[0] !== rank) {
        throw new Error(`Qwen PEFT export rank or projection shape mismatch at ${layerIndex}.${projection}.`);
      }
      const [aValues, bValues] = await Promise.all([
        readF32Tensor(pair.A, aShape, `${layerIndex}.${projection}.lora_A`),
        readF32Tensor(pair.B, bShape, `${layerIndex}.${projection}.lora_B`),
      ]);
      tensors.push(
        {
          name: peftTensorName(layerIndex, layerType, projection, 'A'),
          shape: [aShape[1], aShape[0]],
          data: transposeQwenPeftMatrix(aValues, aShape[0], aShape[1]),
        },
        {
          name: peftTensorName(layerIndex, layerType, projection, 'B'),
          shape: [bShape[1], bShape[0]],
          data: transposeQwenPeftMatrix(bValues, bShape[0], bShape[1]),
        }
      );
    }
  }

  const adapterConfig = {
    base_model_name_or_path: baseModel,
    bias: 'none',
    fan_in_fan_out: false,
    inference_mode: true,
    lora_alpha: alpha,
    lora_dropout: dropout,
    peft_type: 'LORA',
    r: rank,
    target_modules: targetModules,
    task_type: 'CAUSAL_LM',
    use_dora: false,
  };
  return {
    weights: serializeLoRASafetensors(tensors),
    adapterConfig,
    adapterConfigJson: `${JSON.stringify(adapterConfig, null, 2)}\n`,
    tensorCount: tensors.length,
    pairCount: tensors.length / 2,
    elementCount: tensors.reduce((sum, tensor) => sum + tensor.data.length, 0),
    tensorNames: tensors.map((tensor) => tensor.name),
  };
}
