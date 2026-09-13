// Observation only: every WebGPU call delegates to the browser implementation.
// A fresh worker receives this probe and the unchanged current descendant.
{
  const observation = {
    adapter: null, shader: null, compilation: [], pipelines: 0,
    dispatches: [], submissions: 0, mappedBytes: null, mappings: 0,
    destroyedBuffers: 0, destroyedDevices: 0, errors: [], printed: null,
  };
  const failure = (...errors) => {
    observation.errors.push(...errors.map(error => String(error?.stack ?? error)));
    postMessage({ status: 'failed', ...observation });
  };
  console.error = failure;
  console.log = printed => {
    observation.printed = printed;
    // Allow the quine's finally block to finish before reporting cleanup.
    setTimeout(() => postMessage({ status: 'completed', ...observation }), 0);
  };
  addEventListener('error', event => failure(event.error ?? event.message));
  addEventListener('unhandledrejection', event => failure(event.reason));

  if (navigator.gpu) {
    const requestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
    navigator.gpu.requestAdapter = async (...args) => {
      const adapter = await requestAdapter(...args);
      if (!adapter) return adapter;
      const info = adapter.info;
      observation.adapter = {
        vendor: info?.vendor ?? null,
        architecture: info?.architecture ?? null,
        device: info?.device ?? null,
        description: info?.description ?? null,
        isFallbackAdapter: info?.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null,
      };
      const requestDevice = adapter.requestDevice.bind(adapter);
      adapter.requestDevice = async (...deviceArgs) => {
        const device = await requestDevice(...deviceArgs);
        device.addEventListener('uncapturederror', event => failure(event.error));
        const createShaderModule = device.createShaderModule.bind(device);
        device.createShaderModule = descriptor => {
          observation.shader = descriptor.code;
          const module = createShaderModule(descriptor);
          const getCompilationInfo = module.getCompilationInfo.bind(module);
          module.getCompilationInfo = async () => {
            const result = await getCompilationInfo();
            observation.compilation = Array.from(result.messages, message => ({
              type: message.type, message: message.message,
              lineNum: message.lineNum, linePos: message.linePos,
            }));
            return result;
          };
          return module;
        };
        const createPipeline = device.createComputePipelineAsync.bind(device);
        device.createComputePipelineAsync = async descriptor => {
          const pipeline = await createPipeline(descriptor);
          observation.pipelines += 1;
          return pipeline;
        };
        const submit = device.queue.submit.bind(device.queue);
        device.queue.submit = commands => {
          submit(commands);
          observation.submissions += 1;
        };
        return device;
      };
      return adapter;
    };

    const dispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups;
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
      dispatch.apply(this, args);
      observation.dispatches.push(args);
    };
    const getMappedRange = GPUBuffer.prototype.getMappedRange;
    GPUBuffer.prototype.getMappedRange = function (...args) {
      const range = getMappedRange.apply(this, args);
      const words = new Uint32Array(range);
      if (!words[0] || words[0] >= words.length) throw new Error('Invalid mapped GPU length');
      observation.mappings += 1;
      observation.mappedBytes = Array.from(words.subarray(1, words[0] + 1));
      return range;
    };
    const destroyBuffer = GPUBuffer.prototype.destroy;
    GPUBuffer.prototype.destroy = function () {
      destroyBuffer.call(this);
      observation.destroyedBuffers += 1;
    };
    const destroyDevice = GPUDevice.prototype.destroy;
    GPUDevice.prototype.destroy = function () {
      destroyDevice.call(this);
      observation.destroyedDevices += 1;
    };
  }
}
