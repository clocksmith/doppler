const defaultGPUBufferUsage = {
  MAP_READ: 1,
  MAP_WRITE: 2,
  COPY_SRC: 4,
  COPY_DST: 8,
  INDEX: 16,
  VERTEX: 32,
  UNIFORM: 64,
  STORAGE: 128,
  INDIRECT: 256,
  QUERY_RESOLVE: 512,
};

if (typeof globalThis.GPUBufferUsage === 'undefined') {
  globalThis.GPUBufferUsage = defaultGPUBufferUsage;
}

if (typeof globalThis.GPUShaderStage === 'undefined') {
  globalThis.GPUShaderStage = {
    VERTEX: 1,
    FRAGMENT: 2,
    COMPUTE: 4,
  };
}

if (typeof globalThis.GPUMapMode === 'undefined') {
  globalThis.GPUMapMode = {
    READ: 1,
    WRITE: 2,
  };
}
