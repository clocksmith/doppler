
import { selectRuleValue } from '../rules/rule-registry.js';
import { tagBufferDtype } from '../gpu/weight-buffer.js';

export { float16ToFloat32 as f16ToF32 } from '../formats/float16.js';

export async function convertBF16ToF32GPU(srcBuffer, numElements, name) {
  const { runBF16ToF32 } = await import('../gpu/kernels/cast.js');
  const resultTensor = await runBF16ToF32(srcBuffer, [numElements], name);
  return resultTensor.buffer;
}


export function shouldDequantizeToF16(location) {
  const role = location?.role;
  if (!role) {
    throw new Error('Tensor role is required to determine dequantization target.');
  }
  return selectRuleValue('loader', 'weights', 'dequantizeToF16', { role }) === true;
}


function normalizeBufferDtype(locationDtype, outputDtype) {
  const explicit = typeof outputDtype === 'string' ? outputDtype.toLowerCase() : null;
  if (explicit) {
    return explicit;
  }
  const location = typeof locationDtype === 'string' ? locationDtype.toUpperCase() : null;
  if (!location) {
    return null;
  }
  return selectRuleValue('loader', 'weights', 'floatLocationDtype', { locationDtype: location });
}

export function applyBufferLayout(buffer, location, outputDtype = null) {
  // Layout tracking is carried by WeightBuffer. For raw GPUBuffer paths (norms),
  // we still tag runtime dtype so kernels can choose correct weight interpretation.
  const dtype = normalizeBufferDtype(location?.dtype ?? null, outputDtype);
  if (dtype) {
    tagBufferDtype(buffer, dtype);
  }
  return buffer;
}
