/**
 * Kernel Benchmark Functions
 *
 * Individual benchmark implementations for different kernel types.
 * Each benchmark creates test buffers, runs warmup iterations, and measures performance.
 */

/**
 * Benchmark a compute pipeline with given workgroups
 * @param {GPUDevice} device - GPU device
 * @param {GPUComputePipeline} pipeline - Compute pipeline to benchmark
 * @param {GPUBindGroup} bindGroup - Bind group for the pipeline
 * @param {[number, number, number]} workgroups - Number of workgroups [x, y, z]
 * @param {number} warmup - Number of warmup iterations
 * @param {number} iterations - Number of timed iterations
 * @returns {Promise<number>} Average time in milliseconds
 */
export async function benchmarkPipeline(
  device,
  pipeline,
  bindGroup,
  workgroups,
  warmup,
  iterations
) {
  const [wgX, wgY, wgZ] = workgroups;
  if (wgX === 0 || wgY === 0 || wgZ === 0) {
    return Infinity;
  }

  for (let i = 0; i < warmup; i++) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY, wgZ);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  /** @type {number[]} */
  const times = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY, wgZ);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    times.push(performance.now() - start);
  }

  return times.reduce((a, b) => a + b, 0) / times.length;
}

/**
 * Create compute pipeline from shader source
 * @param {GPUDevice} device - GPU device
 * @param {string} shaderSource - WGSL shader source code
 * @param {string} entryPoint - Entry point function name
 * @returns {Promise<GPUComputePipeline>} Compute pipeline
 */
export async function createComputePipeline(
  device,
  shaderSource,
  entryPoint
) {
  const module = device.createShaderModule({ code: shaderSource });
  return await device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint },
  });
}

/**
 * Tune matmul kernel
 * @param {GPUDevice} device - GPU device
 * @param {import('./types.js').InputSizes} inputSizes - Matrix dimensions
 * @param {import('./types.js').WorkgroupSize[]} candidates - Workgroup size candidates
 * @param {number} warmup - Warmup iterations
 * @param {number} iterations - Timed iterations
 * @param {import('./types.js').KernelCapabilities | null} capabilities - Kernel capabilities
 * @returns {Promise<import('./types.js').TuneResult>} Best tuning result
 */
export async function tuneMatmul(
  device,
  inputSizes,
  candidates,
  warmup,
  iterations,
  capabilities
) {
  const { M = 1024, N = 1024, K = 1024 } = inputSizes;

  // Filter to 2D candidates for matmul
  const matmulCandidates = candidates.filter(c => c[1] > 1);

  /** @type {import('./types.js').TuneResult} */
  let best = {
    optimalWorkgroupSize: [16, 16, 1],
    optimalTileSize: 16,
    throughput: 0,
    timeMs: Infinity,
    deviceInfo: capabilities?.adapterInfo,
  };

  // Create test buffers
  const bufferA = device.createBuffer({
    size: M * K * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufferB = device.createBuffer({
    size: K * N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufferC = device.createBuffer({
    size: M * N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Initialize with random data
  const dataA = new Float32Array(M * K);
  const dataB = new Float32Array(K * N);
  for (let i = 0; i < dataA.length; i++) dataA[i] = Math.random();
  for (let i = 0; i < dataB.length; i++) dataB[i] = Math.random();
  device.queue.writeBuffer(bufferA, 0, dataA);
  device.queue.writeBuffer(bufferB, 0, dataB);

  for (const [wgX, wgY] of matmulCandidates) {
    try {
      // Create shader with this workgroup size
      const shader = createMatmulShader(wgX, wgY);
      const pipeline = await createComputePipeline(device, shader, 'main');

      // Create bind group
      const uniformBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const uniformData = new Uint32Array([M, N, K, 0]);
      device.queue.writeBuffer(uniformBuffer, 0, uniformData);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: bufferA } },
          { binding: 2, resource: { buffer: bufferB } },
          { binding: 3, resource: { buffer: bufferC } },
        ],
      });

      // Warmup
      for (let i = 0; i < warmup; i++) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(M / wgX), Math.ceil(N / wgY));
        pass.end();
        device.queue.submit([encoder.finish()]);
      }
      await device.queue.onSubmittedWorkDone();

      // Benchmark
      /** @type {number[]} */
      const times = [];
      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(M / wgX), Math.ceil(N / wgY));
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        times.push(performance.now() - start);
      }

      const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
      const flops = 2 * M * N * K; // multiply-add = 2 ops
      const gflops = (flops / avgTime) / 1e6; // GFLOPS

      if (avgTime < best.timeMs) {
        best = {
          optimalWorkgroupSize: [wgX, wgY, 1],
          optimalTileSize: wgX,
          throughput: gflops,
          timeMs: avgTime,
          deviceInfo: capabilities?.adapterInfo,
        };
      }

      uniformBuffer.destroy();
    } catch (e) {
      // Skip invalid configurations
      continue;
    }
  }

  // Cleanup
  bufferA.destroy();
  bufferB.destroy();
  bufferC.destroy();

  return best;
}

/**
 * Create matmul shader with specified workgroup size
 * @param {number} wgX
 * @param {number} wgY
 * @returns {string}
 */
export function createMatmulShader(wgX, wgY) {
  return `
struct Uniforms {
    M: u32, N: u32, K: u32, _pad: u32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(${wgX}, ${wgY}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= uniforms.M || col >= uniforms.N) { return; }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < uniforms.K; k = k + 1u) {
        sum = sum + A[row * uniforms.K + k] * B[k * uniforms.N + col];
    }
    C[row * uniforms.N + col] = sum;
}`;
}

/**
 * Tune attention kernel
 * @param {GPUDevice} device - GPU device
 * @param {import('./types.js').InputSizes} inputSizes - Input dimensions
 * @param {import('./types.js').WorkgroupSize[]} candidates - Workgroup size candidates
 * @param {number} warmup - Warmup iterations
 * @param {number} iterations - Timed iterations
 * @param {import('./types.js').KernelCapabilities | null} capabilities - Kernel capabilities
 * @returns {Promise<import('./types.js').TuneResult>}
 */
export async function tuneAttention(
  device,
  inputSizes,
  candidates,
  warmup,
  iterations,
  capabilities
) {
  const { seqLen = 2048, numHeads = 32, headDim = 128 } = inputSizes;

  /** @type {import('./types.js').TuneResult} */
  let best = {
    optimalWorkgroupSize: [64, 1, 1],
    optimalTileSize: 64,
    throughput: 0,
    timeMs: Infinity,
    deviceInfo: capabilities?.adapterInfo,
  };

  const attentionCandidates = candidates.filter(c => c[1] === 1);
  if (attentionCandidates.length === 0) {
    return best;
  }

  const maxElements = 2_000_000;
  const totalHeadsRaw = Math.max(1, seqLen * numHeads);
  let benchSeqLen = seqLen;
  let totalHeads = totalHeadsRaw;
  let totalElements = totalHeads * headDim;

  if (totalElements > maxElements) {
    benchSeqLen = Math.max(1, Math.floor(maxElements / (numHeads * headDim)));
    totalHeads = Math.max(1, benchSeqLen * numHeads);
    totalElements = totalHeads * headDim;
  }

  const bufferQ = device.createBuffer({
    size: totalElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufferK = device.createBuffer({
    size: totalElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufferOut = device.createBuffer({
    size: totalHeads * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const dataQ = new Float32Array(totalElements);
  const dataK = new Float32Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    dataQ[i] = Math.random();
    dataK[i] = Math.random();
  }
  device.queue.writeBuffer(bufferQ, 0, dataQ);
  device.queue.writeBuffer(bufferK, 0, dataK);

  for (const [wgX] of attentionCandidates) {
    try {
      const shader = createAttentionShader(wgX);
      const pipeline = await createComputePipeline(device, shader, 'main');

      const uniformBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const uniformData = new Uint32Array([headDim, numHeads, benchSeqLen, 0]);
      device.queue.writeBuffer(uniformBuffer, 0, uniformData);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: bufferQ } },
          { binding: 2, resource: { buffer: bufferK } },
          { binding: 3, resource: { buffer: bufferOut } },
        ],
      });

      const avgTime = await benchmarkPipeline(
        device,
        pipeline,
        bindGroup,
        [totalHeads, 1, 1],
        warmup,
        iterations
      );

      const flops = 2 * totalHeads * headDim;
      const gflops = avgTime > 0 ? (flops / avgTime) / 1e6 : 0;

      if (avgTime < best.timeMs) {
        best = {
          optimalWorkgroupSize: [wgX, 1, 1],
          optimalTileSize: wgX,
          throughput: gflops,
          timeMs: avgTime,
          deviceInfo: capabilities?.adapterInfo,
        };
      }

      uniformBuffer.destroy();
    } catch (e) {
      continue;
    }
  }

  bufferQ.destroy();
  bufferK.destroy();
  bufferOut.destroy();

  return best;
}

/**
 * Create attention shader with specified workgroup size
 * @param {number} wgSize
 * @returns {string}
 */
export function createAttentionShader(wgSize) {
  return `
const WG_SIZE: u32 = ${wgSize}u;

struct Uniforms {
  headDim: u32,
  numHeads: u32,
  seqLen: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;

var<workgroup> shared: array<f32, WG_SIZE>;

@compute @workgroup_size(${wgSize}, 1, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let totalHeads = uniforms.numHeads * uniforms.seqLen;
  let idx = wg_id.x;
  if (idx >= totalHeads) { return; }

  let headDim = uniforms.headDim;
  let offset = idx * headDim;
  let lane = local_id.x;

  var sum: f32 = 0.0;
  var i: u32 = lane;
  loop {
    if (i >= headDim) { break; }
    sum = sum + Q[offset + i] * K[offset + i];
    i = i + WG_SIZE;
  }

  shared[lane] = sum;
  workgroupBarrier();

  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      shared[lane] = shared[lane] + shared[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lane == 0u) {
    Out[idx] = shared[0];
  }
}`;
}

/**
 * Tune softmax kernel
 * @param {GPUDevice} device - GPU device
 * @param {import('./types.js').InputSizes} inputSizes - Input dimensions
 * @param {import('./types.js').WorkgroupSize[]} candidates - Workgroup size candidates
 * @param {number} warmup - Warmup iterations
 * @param {number} iterations - Timed iterations
 * @param {import('./types.js').KernelCapabilities | null} capabilities - Kernel capabilities
 * @returns {Promise<import('./types.js').TuneResult>}
 */
export async function tuneSoftmax(
  device,
  inputSizes,
  candidates,
  warmup,
  iterations,
  capabilities
) {
  const { innerSize = 32000, outerSize = 1 } = inputSizes;

  /** @type {import('./types.js').TuneResult} */
  let best = {
    optimalWorkgroupSize: [256, 1, 1],
    optimalTileSize: 256,
    throughput: 0,
    timeMs: Infinity,
    deviceInfo: capabilities?.adapterInfo,
  };

  const softmaxCandidates = candidates.filter(c => c[1] === 1);
  if (softmaxCandidates.length === 0) {
    return best;
  }

  const totalElements = Math.max(1, innerSize * outerSize);

  const bufferIn = device.createBuffer({
    size: totalElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufferOut = device.createBuffer({
    size: totalElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const dataIn = new Float32Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    dataIn[i] = Math.random();
  }
  device.queue.writeBuffer(bufferIn, 0, dataIn);

  for (const [wgX] of softmaxCandidates) {
    try {
      const shader = createSoftmaxShader(wgX);
      const pipeline = await createComputePipeline(device, shader, 'main');

      const uniformBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const uniformData = new Uint32Array([innerSize, outerSize, 0, 0]);
      device.queue.writeBuffer(uniformBuffer, 0, uniformData);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: bufferIn } },
          { binding: 2, resource: { buffer: bufferOut } },
        ],
      });

      const avgTime = await benchmarkPipeline(
        device,
        pipeline,
        bindGroup,
        [outerSize, 1, 1],
        warmup,
        iterations
      );

      const ops = 2 * totalElements;
      const gops = avgTime > 0 ? (ops / avgTime) / 1e6 : 0;

      if (avgTime < best.timeMs) {
        best = {
          optimalWorkgroupSize: [wgX, 1, 1],
          optimalTileSize: wgX,
          throughput: gops,
          timeMs: avgTime,
          deviceInfo: capabilities?.adapterInfo,
        };
      }

      uniformBuffer.destroy();
    } catch (e) {
      continue;
    }
  }

  bufferIn.destroy();
  bufferOut.destroy();

  return best;
}

/**
 * Create softmax shader with specified workgroup size
 * @param {number} wgSize
 * @returns {string}
 */
export function createSoftmaxShader(wgSize) {
  return `
const WG_SIZE: u32 = ${wgSize}u;

struct Uniforms {
  innerSize: u32,
  outerSize: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, WG_SIZE>;

@compute @workgroup_size(${wgSize}, 1, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let row = wg_id.x;
  if (row >= uniforms.outerSize) { return; }

  let inner = uniforms.innerSize;
  let lane = local_id.x;
  let offset = row * inner;

  var localMax: f32 = -3.402823e+38;
  var i: u32 = lane;
  loop {
    if (i >= inner) { break; }
    localMax = max(localMax, input[offset + i]);
    i = i + WG_SIZE;
  }

  shared[lane] = localMax;
  workgroupBarrier();

  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      shared[lane] = max(shared[lane], shared[lane + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  let rowMax = shared[0];
  var localSum: f32 = 0.0;
  i = lane;
  loop {
    if (i >= inner) { break; }
    localSum = localSum + exp(input[offset + i] - rowMax);
    i = i + WG_SIZE;
  }

  shared[lane] = localSum;
  workgroupBarrier();

  stride = WG_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      shared[lane] = shared[lane] + shared[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  let denom = shared[0];
  i = lane;
  loop {
    if (i >= inner) { break; }
    output[offset + i] = exp(input[offset + i] - rowMax) / denom;
    i = i + WG_SIZE;
  }
}`;
}

/**
 * Tune RMSNorm kernel
 * @param {GPUDevice} device - GPU device
 * @param {import('./types.js').InputSizes} inputSizes - Input dimensions
 * @param {import('./types.js').WorkgroupSize[]} candidates - Workgroup size candidates
 * @param {number} warmup - Warmup iterations
 * @param {number} iterations - Timed iterations
 * @param {import('./types.js').KernelCapabilities | null} capabilities - Kernel capabilities
 * @returns {Promise<import('./types.js').TuneResult>}
 */
export async function tuneRMSNorm(
  device,
  inputSizes,
  candidates,
  warmup,
  iterations,
  capabilities
) {
  const { hiddenSize = 4096, numTokens = 1 } = inputSizes;

  /** @type {import('./types.js').TuneResult} */
  let best = {
    optimalWorkgroupSize: [256, 1, 1],
    optimalTileSize: 256,
    throughput: 0,
    timeMs: Infinity,
    deviceInfo: capabilities?.adapterInfo,
  };

  const rmsCandidates = candidates.filter(c => c[1] === 1);
  if (rmsCandidates.length === 0) {
    return best;
  }

  const totalElements = Math.max(1, hiddenSize * numTokens);

  const bufferIn = device.createBuffer({
    size: totalElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufferWeight = device.createBuffer({
    size: hiddenSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufferOut = device.createBuffer({
    size: totalElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const dataIn = new Float32Array(totalElements);
  const dataWeight = new Float32Array(hiddenSize);
  for (let i = 0; i < totalElements; i++) {
    dataIn[i] = Math.random();
  }
  for (let i = 0; i < hiddenSize; i++) {
    dataWeight[i] = Math.random();
  }
  device.queue.writeBuffer(bufferIn, 0, dataIn);
  device.queue.writeBuffer(bufferWeight, 0, dataWeight);

  for (const [wgX] of rmsCandidates) {
    try {
      const shader = createRMSNormShader(wgX);
      const pipeline = await createComputePipeline(device, shader, 'main');

      const uniformBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const uniformData = new ArrayBuffer(16);
      const uniformView = new DataView(uniformData);
      uniformView.setUint32(0, hiddenSize, true);
      uniformView.setUint32(4, numTokens, true);
      uniformView.setFloat32(8, 1e-5, true);
      uniformView.setUint32(12, 0, true);
      device.queue.writeBuffer(uniformBuffer, 0, uniformData);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: bufferIn } },
          { binding: 2, resource: { buffer: bufferWeight } },
          { binding: 3, resource: { buffer: bufferOut } },
        ],
      });

      const avgTime = await benchmarkPipeline(
        device,
        pipeline,
        bindGroup,
        [numTokens, 1, 1],
        warmup,
        iterations
      );

      const ops = 2 * totalElements;
      const gops = avgTime > 0 ? (ops / avgTime) / 1e6 : 0;

      if (avgTime < best.timeMs) {
        best = {
          optimalWorkgroupSize: [wgX, 1, 1],
          optimalTileSize: wgX,
          throughput: gops,
          timeMs: avgTime,
          deviceInfo: capabilities?.adapterInfo,
        };
      }

      uniformBuffer.destroy();
    } catch (e) {
      continue;
    }
  }

  bufferIn.destroy();
  bufferWeight.destroy();
  bufferOut.destroy();

  return best;
}

/**
 * Create RMSNorm shader with specified workgroup size
 * @param {number} wgSize
 * @returns {string}
 */
export function createRMSNormShader(wgSize) {
  return `
const WG_SIZE: u32 = ${wgSize}u;

struct Uniforms {
  hiddenSize: u32,
  numTokens: u32,
  eps: f32,
  _pad0: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, WG_SIZE>;

@compute @workgroup_size(${wgSize}, 1, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let tokenIdx = wg_id.x;
  if (tokenIdx >= uniforms.numTokens) { return; }

  let size = uniforms.hiddenSize;
  let base = tokenIdx * size;
  let lane = local_id.x;

  var localSumSq: f32 = 0.0;
  var i: u32 = lane;
  loop {
    if (i >= size) { break; }
    let x = input[base + i];
    localSumSq = localSumSq + x * x;
    i = i + WG_SIZE;
  }

  shared[lane] = localSumSq;
  workgroupBarrier();

  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      shared[lane] = shared[lane] + shared[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  let invRms = 1.0 / sqrt(shared[0] / f32(size) + uniforms.eps);
  i = lane;
  loop {
    if (i >= size) { break; }
    output[base + i] = input[base + i] * invRms * weight[i];
    i = i + WG_SIZE;
  }
}`;
}

/**
 * Tune dequantization kernel
 * @param {GPUDevice} device - GPU device
 * @param {import('./types.js').InputSizes} inputSizes - Input dimensions
 * @param {import('./types.js').WorkgroupSize[]} candidates - Workgroup size candidates
 * @param {number} warmup - Warmup iterations
 * @param {number} iterations - Timed iterations
 * @param {import('./types.js').KernelCapabilities | null} capabilities - Kernel capabilities
 * @returns {Promise<import('./types.js').TuneResult>}
 */
export async function tuneDequant(
  device,
  inputSizes,
  candidates,
  warmup,
  iterations,
  capabilities
) {
  const { numBlocks = 1000 } = inputSizes;

  /** @type {import('./types.js').TuneResult} */
  let best = {
    optimalWorkgroupSize: [64, 1, 1],
    optimalTileSize: 64,
    throughput: 0,
    timeMs: Infinity,
    deviceInfo: capabilities?.adapterInfo,
  };

  const dequantCandidates = candidates.filter(c => c[1] === 1);
  if (dequantCandidates.length === 0) {
    return best;
  }

  const numElements = Math.max(1, numBlocks * 256);

  const bufferIn = device.createBuffer({
    size: numElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufferOut = device.createBuffer({
    size: numElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const dataIn = new Uint32Array(numElements);
  for (let i = 0; i < numElements; i++) {
    dataIn[i] = i & 0xffff;
  }
  device.queue.writeBuffer(bufferIn, 0, dataIn);

  for (const [wgX] of dequantCandidates) {
    try {
      const shader = createDequantShader(wgX);
      const pipeline = await createComputePipeline(device, shader, 'main');

      const uniformBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const uniformData = new ArrayBuffer(16);
      const uniformView = new DataView(uniformData);
      uniformView.setUint32(0, numElements, true);
      uniformView.setFloat32(4, 0.01, true);
      uniformView.setUint32(8, 0, true);
      uniformView.setUint32(12, 0, true);
      device.queue.writeBuffer(uniformBuffer, 0, uniformData);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: bufferIn } },
          { binding: 2, resource: { buffer: bufferOut } },
        ],
      });

      const workgroups = Math.ceil(numElements / wgX);
      const avgTime = await benchmarkPipeline(
        device,
        pipeline,
        bindGroup,
        [workgroups, 1, 1],
        warmup,
        iterations
      );

      const ops = numElements;
      const gops = avgTime > 0 ? (ops / avgTime) / 1e6 : 0;

      if (avgTime < best.timeMs) {
        best = {
          optimalWorkgroupSize: [wgX, 1, 1],
          optimalTileSize: wgX,
          throughput: gops,
          timeMs: avgTime,
          deviceInfo: capabilities?.adapterInfo,
        };
      }

      uniformBuffer.destroy();
    } catch (e) {
      continue;
    }
  }

  bufferIn.destroy();
  bufferOut.destroy();

  return best;
}

/**
 * Create dequant shader with specified workgroup size
 * @param {number} wgSize
 * @returns {string}
 */
export function createDequantShader(wgSize) {
  return `
const WG_SIZE: u32 = ${wgSize}u;

struct Uniforms {
  count: u32,
  scale: f32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${wgSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uniforms.count) { return; }
  output[idx] = f32(input[idx]) * uniforms.scale;
}`;
}

/**
 * Generic tuning for unknown kernels - returns sensible defaults
 * @param {import('./types.js').KernelCapabilities | null} capabilities - Kernel capabilities
 * @returns {import('./types.js').TuneResult}
 */
export function tuneGeneric(capabilities) {
  return {
    optimalWorkgroupSize: [256, 1, 1],
    optimalTileSize: 256,
    throughput: 0,
    timeMs: 0,
    deviceInfo: capabilities?.adapterInfo,
  };
}
