
import { selectRuleValue } from '../rules/rule-registry.js';
import { tagBufferDtype } from '../gpu/weight-buffer.js';


export function f16ToF32(h) {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;

  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0;
    const f = mant / 1024 * Math.pow(2, -14);
    return sign ? -f : f;
  }
  if (exp === 31) {
    return mant ? NaN : (sign ? -Infinity : Infinity);
  }

  const f = (1 + mant / 1024) * Math.pow(2, exp - 15);
  return sign ? -f : f;
}


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
