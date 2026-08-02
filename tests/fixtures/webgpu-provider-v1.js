export const globals = Object.freeze({
  GPUBufferUsage: Object.freeze({ STORAGE: 1 }),
  GPUShaderStage: Object.freeze({ COMPUTE: 2 }),
  GPUMapMode: Object.freeze({ READ: 4 }),
  GPUTextureUsage: Object.freeze({ STORAGE_BINDING: 8 }),
});

export function create() {
  return {
    async requestAdapter() {
      return Object.freeze({ label: 'doppler-provider-v1-test-adapter' });
    },
  };
}
