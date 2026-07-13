import { parseSafetensorsHeader } from '../../formats/safetensors/types.js';
import { uploadData } from '../../memory/buffer-pool.js';

const ATTENTION_PROJECTIONS = Object.freeze(['q_proj', 'k_proj', 'v_proj', 'o_proj']);
const MLP_PROJECTIONS = Object.freeze(['gate_proj', 'up_proj', 'down_proj']);
const SUPPORTED_PROJECTIONS = new Set([...ATTENTION_PROJECTIONS, ...MLP_PROJECTIONS]);
const PROJECTION_ORDER = new Map(
  [...ATTENTION_PROJECTIONS, ...MLP_PROJECTIONS].map((name, index) => [name, index])
);

function asArrayBuffer(data) {
  if (data instanceof ArrayBuffer) return data;
  if (ArrayBuffer.isView(data)) {
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }
  throw new Error('Qwen PEFT adapter import requires ArrayBuffer or typed-array weights.');
}

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

function normalizeLayerTypes(value) {
  if (!Array.isArray(value) || value.length < 1) {
    throw new Error('Qwen PEFT adapter import requires an explicit layerTypes array.');
  }
  return value.map((type, index) => {
    if (type !== 'linear_attention' && type !== 'full_attention') {
      throw new Error(`Qwen PEFT layerTypes[${index}] has unsupported type "${String(type)}".`);
    }
    return type;
  });
}

function normalizeTargetModules(value) {
  if (!Array.isArray(value) || value.length < 1) {
    throw new Error('Qwen PEFT adapter import requires targetModules.');
  }
  const modules = [...new Set(value.map((item) => String(item)))];
  for (const moduleName of modules) {
    if (!SUPPORTED_PROJECTIONS.has(moduleName)) {
      throw new Error(`Qwen PEFT adapter target module "${moduleName}" is unsupported.`);
    }
  }
  return modules;
}

function parseTensorName(name) {
  const match = String(name).match(
    /(?:^|\.)layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_([AB])(?:\.[^.]+)?\.weight$/
  );
  if (!match) return null;
  return {
    layerIndex: Number(match[1]),
    branch: match[2],
    projection: match[3],
    kind: match[4] === 'A' ? 'a' : 'b',
  };
}

function float16ToFloat32(value) {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) {
    return fraction === 0 ? (sign < 0 ? -0 : 0) : sign * 2 ** -14 * (fraction / 1024);
  }
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function bfloat16ToFloat32(value) {
  const scratch = new DataView(new ArrayBuffer(4));
  scratch.setUint32(0, value << 16, true);
  return scratch.getFloat32(0, true);
}

function decodeTensor(buffer, tensor) {
  const shape = Array.isArray(tensor.shape) ? tensor.shape.map(Number) : [];
  if (shape.length !== 2 || shape.some((value) => !Number.isInteger(value) || value < 1)) {
    throw new Error(`Qwen PEFT tensor ${tensor.name} must have a positive rank-two shape.`);
  }
  const elementCount = shape[0] * shape[1];
  const bytesPerElement = tensor.dtypeOriginal === 'F32' ? 4 : 2;
  if (!['F32', 'F16', 'BF16'].includes(tensor.dtypeOriginal)) {
    throw new Error(`Qwen PEFT tensor ${tensor.name} has unsupported dtype ${tensor.dtypeOriginal}.`);
  }
  if (tensor.size !== elementCount * bytesPerElement) {
    throw new Error(
      `Qwen PEFT tensor ${tensor.name} byte size mismatch: expected ${elementCount * bytesPerElement}, got ${tensor.size}.`
    );
  }
  if (tensor.offset < 0 || tensor.offset + tensor.size > buffer.byteLength) {
    throw new Error(`Qwen PEFT tensor ${tensor.name} lies outside the safetensors payload.`);
  }
  const view = new DataView(buffer);
  const values = new Float32Array(elementCount);
  for (let index = 0; index < elementCount; index += 1) {
    const offset = tensor.offset + (index * bytesPerElement);
    if (tensor.dtypeOriginal === 'F32') {
      values[index] = view.getFloat32(offset, true);
    } else if (tensor.dtypeOriginal === 'F16') {
      values[index] = float16ToFloat32(view.getUint16(offset, true));
    } else {
      values[index] = bfloat16ToFloat32(view.getUint16(offset, true));
    }
  }
  return { shape, values };
}

function transpose(values, rows, columns) {
  const output = new Float32Array(values.length);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      output[(column * rows) + row] = values[(row * columns) + column];
    }
  }
  return output;
}

function canonicalTensorName(parsed) {
  const branch = parsed.branch === 'self_attn' ? 'self_attn' : 'mlp';
  return `layers.${parsed.layerIndex}.${branch}.${parsed.projection}.lora_${parsed.kind}`;
}

function expectedProjections(layerType, targetModules) {
  return [
    ...(layerType === 'full_attention' ? ATTENTION_PROJECTIONS : []),
    ...MLP_PROJECTIONS,
  ].filter((name) => targetModules.includes(name));
}

function validateTopology(tensors, config) {
  const pairs = new Map();
  for (const tensor of tensors) {
    const key = `${tensor.layerIndex}.${tensor.projection}`;
    const pair = pairs.get(key) ?? {};
    if (pair[tensor.kind]) {
      throw new Error(`Qwen PEFT adapter contains duplicate tensor ${tensor.canonicalName}.`);
    }
    pair[tensor.kind] = tensor;
    pairs.set(key, pair);
  }

  let expectedPairCount = 0;
  for (let layerIndex = 0; layerIndex < config.layerTypes.length; layerIndex += 1) {
    for (const projection of expectedProjections(config.layerTypes[layerIndex], config.targetModules)) {
      expectedPairCount += 1;
      const key = `${layerIndex}.${projection}`;
      const pair = pairs.get(key);
      if (!pair?.a || !pair?.b) {
        throw new Error(`Qwen PEFT adapter is missing a complete A/B pair for ${key}.`);
      }
      if (pair.a.shape[1] !== config.rank || pair.b.shape[0] !== config.rank) {
        throw new Error(`Qwen PEFT adapter internal rank mismatch for ${key}.`);
      }
    }
  }
  if (pairs.size !== expectedPairCount) {
    const expected = new Set();
    for (let layerIndex = 0; layerIndex < config.layerTypes.length; layerIndex += 1) {
      for (const projection of expectedProjections(config.layerTypes[layerIndex], config.targetModules)) {
        expected.add(`${layerIndex}.${projection}`);
      }
    }
    const extras = [...pairs.keys()].filter((key) => !expected.has(key));
    throw new Error(`Qwen PEFT adapter contains unexpected projection pairs: ${extras.join(', ')}.`);
  }
  return { pairCount: pairs.size };
}

export function parseQwenPeftAdapterSafetensors(data, options) {
  const buffer = asArrayBuffer(data);
  const config = {
    rank: positiveInteger(options?.rank ?? options?.r, 'Qwen PEFT rank'),
    alpha: finiteNumber(options?.alpha ?? options?.lora_alpha, 'Qwen PEFT alpha'),
    targetModules: normalizeTargetModules(options?.targetModules ?? options?.target_modules),
    layerTypes: normalizeLayerTypes(options?.layerTypes),
  };
  const parsedHeader = parseSafetensorsHeader(buffer);
  const tensors = [];
  for (const tensor of parsedHeader.tensors) {
    const parsed = parseTensorName(tensor.name);
    if (!parsed) {
      throw new Error(`Qwen PEFT adapter contains unrecognized tensor ${tensor.name}.`);
    }
    if (parsed.layerIndex >= config.layerTypes.length) {
      throw new Error(`Qwen PEFT tensor ${tensor.name} exceeds the declared layerTypes array.`);
    }
    if (!config.targetModules.includes(parsed.projection)) {
      throw new Error(`Qwen PEFT tensor ${tensor.name} is outside targetModules.`);
    }
    const isAttention = ATTENTION_PROJECTIONS.includes(parsed.projection);
    if ((isAttention && parsed.branch !== 'self_attn') || (!isAttention && parsed.branch !== 'mlp')) {
      throw new Error(`Qwen PEFT tensor ${tensor.name} has the wrong model branch.`);
    }
    if (isAttention && config.layerTypes[parsed.layerIndex] !== 'full_attention') {
      throw new Error(`Qwen PEFT tensor ${tensor.name} targets attention on a linear-attention layer.`);
    }
    const decoded = decodeTensor(buffer, tensor);
    const [rows, columns] = decoded.shape;
    if ((parsed.kind === 'a' && rows !== config.rank)
      || (parsed.kind === 'b' && columns !== config.rank)) {
      throw new Error(`Qwen PEFT tensor ${tensor.name} does not match declared rank ${config.rank}.`);
    }
    tensors.push({
      sourceName: tensor.name,
      canonicalName: canonicalTensorName(parsed),
      layerIndex: parsed.layerIndex,
      branch: parsed.branch,
      projection: parsed.projection,
      kind: parsed.kind,
      sourceDtype: tensor.dtypeOriginal,
      sourceShape: decoded.shape,
      shape: [columns, rows],
      data: transpose(decoded.values, rows, columns),
    });
  }
  tensors.sort((left, right) => (
    left.layerIndex - right.layerIndex
    || PROJECTION_ORDER.get(left.projection) - PROJECTION_ORDER.get(right.projection)
    || left.kind.localeCompare(right.kind)
  ));
  const topology = validateTopology(tensors, config);
  return {
    ...config,
    scale: config.alpha / config.rank,
    tensors,
    tensorCount: tensors.length,
    pairCount: topology.pairCount,
    elementCount: tensors.reduce((sum, tensor) => sum + tensor.data.length, 0),
  };
}

function targetAdapterPair(layer, tensor) {
  const shortName = tensor.projection.replace(/_proj$/, '');
  if (tensor.branch === 'self_attn') {
    return layer.inputs?.attention?.lora?.[shortName];
  }
  return layer.inputs?.mlp?.lora?.[shortName];
}

function sameShape(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Number(value) === Number(right[index]));
}

export function buildQwenPeftAdapterUploadPlan(layers, adapter) {
  if (!Array.isArray(layers) || layers.length !== adapter?.layerTypes?.length) {
    throw new Error('Qwen PEFT upload layer count does not match the imported adapter.');
  }
  const plan = [];
  for (let index = 0; index < layers.length; index += 1) {
    if (layers[index]?.type !== adapter.layerTypes[index]) {
      throw new Error(`Qwen PEFT upload layer ${index} type mismatch.`);
    }
  }
  for (const entry of adapter.tensors) {
    const pair = targetAdapterPair(layers[entry.layerIndex], entry);
    const target = pair?.[entry.kind === 'a' ? 'A' : 'B'];
    if (!target?.buffer) {
      throw new Error(`Qwen PEFT upload target is missing for ${entry.canonicalName}.`);
    }
    if (target.dtype !== 'f32') {
      throw new Error(`Qwen PEFT upload target ${entry.canonicalName} must use f32 LoRA storage.`);
    }
    if (!sameShape(target.shape, entry.shape)) {
      throw new Error(
        `Qwen PEFT upload shape mismatch for ${entry.canonicalName}: expected [${entry.shape}], got [${target.shape}].`
      );
    }
    plan.push({
      sourceName: entry.sourceName,
      canonicalName: entry.canonicalName,
      tensor: target,
      data: entry.data,
    });
  }
  return plan;
}

export function uploadQwenPeftAdapterToLayers(layers, adapter) {
  const plan = buildQwenPeftAdapterUploadPlan(layers, adapter);
  for (const entry of plan) uploadData(entry.tensor.buffer, entry.data);
  return {
    tensorCount: plan.length,
    elementCount: plan.reduce((sum, entry) => sum + entry.data.length, 0),
    canonicalNames: plan.map((entry) => entry.canonicalName),
  };
}
