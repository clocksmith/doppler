/**
 * GPU submit latency probe.
 *
 * Dispatches a trivial compute shader, submits, waits for the GPU fence
 * (onSubmittedWorkDone) plus a staging-buffer readback (mapAsync), and
 * returns the wall-clock roundtrip in milliseconds.
 *
 * The result is used at session init to decide whether the batched-GPU
 * decode path is viable or whether single-token decode should be preferred.
 */

const PROBE_SHADER = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
  out[0] = 1u;
}
`;

/**
 * Run a single submit+readback roundtrip and return the elapsed time in ms.
 * Returns `null` if the probe cannot run (e.g. device lost, buffer failure).
 *
 * @param {GPUDevice} device
 * @returns {Promise<number | null>}
 */
export async function probeSubmitLatency(device) {
  if (!device) return null;

  let gpuBuffer = null;
  let stagingBuffer = null;

  try {
    gpuBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    stagingBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const module = device.createShaderModule({ code: PROBE_SHADER });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: gpuBuffer } }],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(gpuBuffer, 0, stagingBuffer, 0, 4);

    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    stagingBuffer.unmap();
    const elapsed = performance.now() - start;

    return elapsed;
  } catch {
    return null;
  } finally {
    stagingBuffer?.destroy();
    gpuBuffer?.destroy();
  }
}
