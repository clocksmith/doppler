import { validateFP8Policy } from '../../config/schema/fp8.schema.js';
import { validateFP8StorageDescriptor, fp8StorageBytes } from './fp8-storage.js';

/** @type {typeof import('./fp8-contract.js').validateFP8ManifestContract} */
export function validateFP8ManifestContract(manifest) {
  const errors = [];
  const info = manifest.quantizationInfo;
  let policy = null;
  if (info?.fp8 != null || info?.weights === 'fp8e4' || info?.weights === 'fp8e5') {
    try {
      policy = validateFP8Policy(info.fp8, 'manifest.quantizationInfo.fp8');
      if ((info.weights === 'fp8e4' && policy.encoding !== 'e4m3fn')
        || (info.weights === 'fp8e5' && policy.encoding !== 'e5m2')
        || !['fp8e4', 'fp8e5', 'f16', 'f32'].includes(String(info.weights)) || info.compute !== 'f32'
        || !['f16', 'f32'].includes(String(info.embeddings)) || !['f16', 'f32'].includes(String(info.lmHead ?? info.embeddings))) {
        throw new Error('FP8 manifest requires matching encoding, explicit F32 compute and dense embedding/head storage.');
      }
    } catch (error) { errors.push(error.message); }
  }
  for (const [name, tensor] of Object.entries(manifest.tensors ?? {})) {
    const packed = tensor.dtype === 'FP8_E4M3FN' || tensor.dtype === 'FP8_E5M2';
    const storage = /** @type {{ packing?: string } | undefined} */ (tensor.storage);
    if (!packed && storage?.packing !== 'fp8-block-v1') continue;
    try {
      if (!policy || !packed || tensor.role !== 'matmul' || tensor.layout !== 'row'
        || info?.weights !== (policy.encoding === 'e4m3fn' ? 'fp8e4' : 'fp8e5')
        || tensor.dtype !== (policy.encoding === 'e4m3fn' ? 'FP8_E4M3FN' : 'FP8_E5M2')) {
        throw new Error(`FP8 tensor ${name} lacks a matching packed matmul contract.`);
      }
      const descriptor = validateFP8StorageDescriptor(tensor.storage);
      if (descriptor.encoding !== policy.encoding || JSON.stringify(descriptor.blockShape) !== JSON.stringify(policy.blockShape)
        || !tensor.shape || tensor.size !== fp8StorageBytes(tensor.shape, descriptor)) {
        throw new Error(`FP8 tensor ${name} block geometry/encoding/size differs from its manifest.`);
      }
    } catch (error) { errors.push(error.message); }
  }
  return errors;
}
