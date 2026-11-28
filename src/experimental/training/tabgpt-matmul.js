import { compileShader, loadShaderSource } from '../../gpu/kernels/shader-cache.js';

// Resolve every static product once. Aligned products keep a 4x4 output tile
// in registers; scalar tails use Doppler's bounds-checked general FP32 kernels.
/**
 * @param {GPUDevice} device
 * @param {(size: number, usage: GPUBufferUsageFlags, label: string) => GPUBuffer} allocateBuffer
 * @returns {Promise<import('./tabgpt-matmul.js').TabGPTMatmul>}
 */
export async function createTabGPTMatmul(device, allocateBuffer) {
  const files = [
    'src/gpu/kernels/matmul_f32.wgsl',
    'src/gpu/kernels/backward/matmul_transpose_a.wgsl',
    'src/gpu/kernels/training/tabgpt_matmul_register.wgsl',
  ];
  const modules = await Promise.all(files.map(async (file) =>
    compileShader(device, await loadShaderSource(file), file)));
  const general = await Promise.all(modules.slice(0, 2).map((module, i) =>
    device.createComputePipelineAsync({
      label: files[i], layout: 'auto', compute: { module, entryPoint: 'main' },
    })));
  /** @type {Map<string, ReturnType<typeof makePlan>>} */
  const plans = new Map();

  /** @param {number} m @param {number} n @param {number} k @param {''|'a'|'b'} transpose */
  function makePlan(m, n, k, transpose) {
    const aligned = m % 4 === 0 && n % 4 === 0 && k % 4 === 0;
    const x = aligned ? Math.ceil(n / 64) : Math.ceil(m / 16);
    const y = aligned ? Math.ceil(m / 32) : Math.ceil(n / 16);
    if (Math.max(x, y) > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new RangeError(`TabGPT matmul requires ${x} by ${y} workgroups, exceeding the device limit.`);
    }
    if (aligned) {
      const pipeline = device.createComputePipeline({
        label: files[2], layout: 'auto',
        compute: { module: modules[2], entryPoint: 'main', constants: {
          M: m, N: n, K: k, TRANSPOSE_A: Number(transpose === 'a'), TRANSPOSE_B: Number(transpose === 'b'),
        } },
      });
      return { x, y, pipeline, uniform: null };
    }
    const data = new Uint32Array(8);
    data.set([m, n, k]);
    new Float32Array(data.buffer)[3] = 1;
    data[4] = Number(transpose === 'b');
    const uniform = allocateBuffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'matmul-shape');
    device.queue.writeBuffer(uniform, 0, data);
    return { x, y, pipeline: general[Number(transpose === 'a')], uniform };
  }

  return function matmul(a, b, out, m, n, k, transpose = '') {
    const key = [m, n, k, transpose].join('/');
    let plan = plans.get(key);
    if (!plan) {
      plan = makePlan(m, n, k, transpose);
      plans.set(key, plan);
    }
    const { x, y, pipeline, uniform } = plan;
    const buffers = uniform ? [uniform, a, b, out] : [a, b, out];
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((value, binding) => ({
        binding, resource: 'buffer' in value ? value : { buffer: value },
      })),
    });
    return (pass) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(x, y);
    };
  };
}
