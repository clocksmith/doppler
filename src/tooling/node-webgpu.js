function hasNavigatorGpu() {
  return typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator?.gpu;
}

function hasGpuEnums() {
  return typeof globalThis.GPUBufferUsage !== 'undefined' && typeof globalThis.GPUShaderStage !== 'undefined';
}

function setGlobalIfMissing(name, value) {
  if (value === undefined || value === null) return;
  if (globalThis[name] !== undefined) return;
  Object.defineProperty(globalThis, name, {
    value,
    writable: true,
    configurable: true,
    enumerable: false,
  });
}

function installNavigatorGpu(gpu) {
  if (!gpu || typeof gpu.requestAdapter !== 'function') return false;
  if (typeof globalThis.navigator === 'undefined') {
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu },
      writable: true,
      configurable: true,
      enumerable: false,
    });
    return true;
  }

  if (!globalThis.navigator.gpu) {
    Object.defineProperty(globalThis.navigator, 'gpu', {
      value: gpu,
      writable: true,
      configurable: true,
      enumerable: false,
    });
  }
  return true;
}

function resolveGpuFromModule(mod) {
  if (!mod) return null;

  const fromModule = mod.gpu || mod.webgpu || mod.default?.gpu || mod.default?.webgpu;
  if (fromModule && typeof fromModule.requestAdapter === 'function') {
    return fromModule;
  }

  const factory = mod.create || mod.default?.create;
  if (typeof factory === 'function') {
    let created = null;
    try {
      created = factory();
    } catch {
      created = null;
    }
    if (created) {
      if (typeof created.requestAdapter === 'function') {
        return created;
      }
      if (created.gpu && typeof created.gpu.requestAdapter === 'function') {
        return created.gpu;
      }
    }
  }

  if (mod.default && typeof mod.default.requestAdapter === 'function') {
    return mod.default;
  }

  return null;
}

export async function bootstrapNodeWebGPU() {
  if (hasNavigatorGpu() && hasGpuEnums()) {
    return true;
  }

  let mod;
  try {
    mod = await import('webgpu');
  } catch {
    return false;
  }

  const gpu = resolveGpuFromModule(mod);
  if (!installNavigatorGpu(gpu)) {
    return false;
  }

  setGlobalIfMissing('GPUBufferUsage', mod.GPUBufferUsage || mod.default?.GPUBufferUsage);
  setGlobalIfMissing('GPUShaderStage', mod.GPUShaderStage || mod.default?.GPUShaderStage);
  setGlobalIfMissing('GPUMapMode', mod.GPUMapMode || mod.default?.GPUMapMode);
  setGlobalIfMissing('GPUTextureUsage', mod.GPUTextureUsage || mod.default?.GPUTextureUsage);

  return hasNavigatorGpu() && hasGpuEnums();
}
