import { createTensor } from '../gpu/tensor.js';
import { getBuffer, getWeightDtype, isGpuBufferInstance } from '../gpu/weight-buffer.js';

export async function loadGpuTensor(loader, name, silent = false) {
  const loaded = await loader._loadTensor(name, true, silent);
  if (!loaded) return null;
  const location = loader.tensorLocations.get(name);
  if (!location || !Array.isArray(location.shape)) {
    throw new Error(`GPU tensor "${name}" is missing its manifest shape.`);
  }
  const buffer = getBuffer(loaded);
  if (!isGpuBufferInstance(buffer)) {
    throw new Error(`GPU tensor "${name}" did not resolve to a GPU buffer.`);
  }
  const dtype = getWeightDtype(loaded);
  if (dtype !== 'f16' && dtype !== 'f32') {
    throw new Error(`GPU tensor "${name}" requires f16 or f32 data, got ${String(dtype)}.`);
  }
  return createTensor(buffer, dtype, location.shape, name);
}
