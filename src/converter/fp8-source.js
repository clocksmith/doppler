import { validateFP8Policy } from '../config/schema/fp8.schema.js';
import { float8ToFloat32 } from '../formats/float8.js';
import { createFP8StorageDescriptor, fp8StorageBytes, validateFP8StorageBytes } from '../formats/rdrr/fp8-storage.js';

/** @type {typeof import('./fp8-source.js').fp8SourceEncoding} */
export function fp8SourceEncoding(dtype) {
  if (dtype === 'F8_E4M3') return 'e4m3fn';
  if (dtype === 'F8_E5M2') return 'e5m2';
  return null;
}

/** @type {typeof import('./fp8-source.js').prepareFP8SourceTensors} */
export function prepareFP8SourceTensors(tensors, policy, modelConfig) {
  const fp8 = tensors.filter(tensor => typeof tensor.dtype === 'string' && tensor.dtype.startsWith('F8_'));
  const companions = new Map();
  if (!fp8.length) return { tensors, companions };
  const resolved = validateFP8Policy(policy, 'converter.quantization.fp8');
  const sourceQuant = modelConfig?.quantization_config;
  if ((sourceQuant?.quant_method != null && sourceQuant.quant_method !== 'fp8')
    || (sourceQuant?.fmt != null && sourceQuant.fmt !== (resolved.encoding === 'e4m3fn' ? 'e4m3' : 'e5m2'))
    || sourceQuant?.scale_fmt != null) throw new Error('FP8 source quantization encoding/scaling metadata is unsupported or conflicts with the explicit policy.');
  if (sourceQuant && resolved.scaleMode === 'identity') throw new Error('FP8 identity scales cannot ignore source quantization_config.');
  if (sourceQuant?.weight_block_size
    && JSON.stringify(sourceQuant.weight_block_size) !== JSON.stringify(resolved.blockShape)) {
    throw new Error('FP8 source weight_block_size differs from explicit blockShape.');
  }
  const byName = new Map(tensors.map(tensor => [tensor.name, tensor]));
  const consumed = new Set();
  for (const tensor of fp8) {
    if (fp8SourceEncoding(tensor.dtype) !== resolved.encoding || tensor.shape.length !== 2
      || (tensor.layout != null && tensor.layout !== 'row')) {
      throw new Error(`FP8 source ${tensor.name} requires the declared encoding and row-major [N,K] shape.`);
    }
    if (resolved.scaleMode === 'companion') {
      const companion = byName.get(tensor.name + resolved.scaleSuffix);
      const expected = tensor.shape.map((value, index) => Math.ceil(value / resolved.blockShape[index]));
      if (!companion || companion.dtype !== 'F32' || JSON.stringify(companion.shape) !== JSON.stringify(expected)) {
        throw new Error(`FP8 source ${tensor.name} requires its explicit F32 block-scale companion ${tensor.name + resolved.scaleSuffix}.`);
      }
      companions.set(tensor.name, companion);
      consumed.add(companion.name);
    }
  }
  return { tensors: tensors.filter(tensor => !consumed.has(tensor.name)), companions };
}

/** @type {typeof import('./fp8-source.js').readFP8CompanionScales} */
export async function readFP8CompanionScales(tensor, companion, policy, read) {
  if (!fp8SourceEncoding(tensor.dtype)) return null;
  const resolved = validateFP8Policy(policy);
  if (resolved.scaleMode === 'identity') return null;
  if (!companion) throw new Error(`FP8 source ${tensor.name} has no admitted scale companion.`);
  const data = await read(companion);
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const expected = companion.shape[0] * companion.shape[1];
  if (bytes.byteLength !== expected * 4) throw new Error('FP8 scale companion byte length differs from its shape.');
  const values = new Float32Array(expected);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < expected; index++) values[index] = view.getFloat32(index * 4, true);
  return values;
}

/** @type {typeof import('./fp8-source.js').packFP8Source} */
export function packFP8Source(bytes, shape, policy, scales) {
  const storage = createFP8StorageDescriptor(policy);
  const quantized = new Uint8Array(fp8StorageBytes(shape, storage));
  if (!(bytes instanceof Uint8Array) || bytes.length !== shape[0] * shape[1]) throw new Error('FP8 source byte length differs from [N,K].');
  const [blockN, blockK] = storage.blockShape;
  const blocksN = Math.ceil(shape[0] / blockN), blocksK = Math.ceil(shape[1] / blockK);
  if ((policy.scaleMode === 'identity' && scales !== null)
    || (policy.scaleMode === 'companion' && (!(scales instanceof Float32Array) || scales.length !== blocksN * blocksK))) {
    throw new Error('FP8 scales must exactly match the explicit source scale mode and block geometry.');
  }
  const view = new DataView(quantized.buffer);
  for (let block = 0; block < blocksN * blocksK; block++) {
    const base = block * storage.blockBytes;
    view.setFloat32(base, policy.scaleMode === 'identity' ? 1 : /** @type {Float32Array} */ (scales)[block], true);
    for (let n = 0; n < blockN; n++) for (let k = 0; k < blockK; k++) {
      const row = Math.floor(block / blocksK) * blockN + n;
      const column = (block % blocksK) * blockK + k;
      if (row < shape[0] && column < shape[1]) quantized[base + 4 + n * blockK + k] = bytes[row * shape[1] + column];
    }
  }
  validateFP8StorageBytes(quantized, shape, storage);
  return { quantized, storage };
}

/** @type {typeof import('./fp8-source.js').materializeFP8Source} */
export function materializeFP8Source(bytes, shape, policy, scales) {
  packFP8Source(bytes, shape, policy, scales);
  const output = new Float32Array(bytes.length);
  const blocksK = Math.ceil(shape[1] / policy.blockShape[1]);
  for (let n = 0; n < shape[0]; n++) for (let k = 0; k < shape[1]; k++) {
    const block = Math.floor(n / policy.blockShape[0]) * blocksK + Math.floor(k / policy.blockShape[1]);
    const scale = policy.scaleMode === 'identity' ? 1 : /** @type {Float32Array} */ (scales)[block];
    output[n * shape[1] + k] = Math.fround(float8ToFloat32(bytes[n * shape[1] + k], policy.encoding) * scale);
  }
  return output;
}
