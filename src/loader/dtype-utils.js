

import { getDevice } from '../gpu/device.js';
import { isTraceEnabled, log, trace as debugTrace } from '../debug/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';


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
  debugTrace.loader(`[BF16→F32] Importing cast.js...`);
  const castModule = await import('../gpu/kernels/cast.js');
  debugTrace.loader(`[BF16→F32] castModule keys:`, Object.keys(castModule));
  const { runBF16ToF32 } = castModule;
  debugTrace.loader(`[BF16→F32] runBF16ToF32 type: ${typeof runBF16ToF32}`);
  const resultTensor = await runBF16ToF32(srcBuffer, [numElements], name);
  debugTrace.loader(`[BF16→F32] runBF16ToF32 returned, result.size=${resultTensor.buffer?.size}`);

  // Debug: Verify conversion produced non-zero values
  const shouldCheckEmbed = isTraceEnabled('loader') &&
    name.includes('embed') &&
    name.includes('embed_tokens');
  if (shouldCheckEmbed) {
    try {
      debugTrace.loader(`[BF16→F32] Checking embed buffer for non-zeros...`);
      const device = getDevice();
      const sampleSize = Math.min(1024, resultTensor.buffer.size);
      debugTrace.loader(`[BF16→F32] Creating staging buffer size=${sampleSize}`);
      const stagingBuffer = device.createBuffer({
        size: sampleSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      debugTrace.loader(`[BF16→F32] Copying to staging buffer...`);
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(resultTensor.buffer, 0, stagingBuffer, 0, sampleSize);
      device.queue.submit([encoder.finish()]);
      debugTrace.loader(`[BF16→F32] Mapping staging buffer...`);
      await stagingBuffer.mapAsync(GPUMapMode.READ);
      debugTrace.loader(`[BF16→F32] Reading data...`);
      const data = new Float32Array(stagingBuffer.getMappedRange().slice(0));
      stagingBuffer.unmap();
      stagingBuffer.destroy();
      const nonZero = Array.from(data).filter(x => x !== 0);
      const nanCount = data.filter(x => !Number.isFinite(x)).length;
      debugTrace.loader(`[BF16→F32] nonZero=${nonZero.length}/${data.length}, nan=${nanCount}, sample=[${nonZero.slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
    } catch (err) {
      log.error('Loader', 'BF16→F32 embed buffer check error:',  (err).message);
    }
  }

  return resultTensor.buffer;
}


export function shouldDequantizeToF16(location) {
  const role = location?.role;
  if (!role) {
    throw new Error('Tensor role is required to determine dequantization target.');
  }
  return selectRuleValue('loader', 'weights', 'dequantizeToF16', { role }) === true;
}


export function applyBufferLayout(buffer, _location) {
  // Note: WeakMap layout tracking removed - layout is stored in WeightBuffer
  // For non-matmul weights (norms), layout doesn't affect kernel selection
  return buffer;
}
