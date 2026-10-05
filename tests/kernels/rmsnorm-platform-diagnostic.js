/** Numerical experiments only. Candidate WGSL is not a shipped runtime implementation. */
export function buildCompensatedRMSNormDiagnostic(source) {
  const start = source.indexOf('fn main(');
  const end = source.indexOf('// Small Hidden Size Entry Point', start);
  if (start < 0 || end < 0) throw Error('Canonical normalization main boundary not found');
  let main = source.slice(start, end);
  const replace = (from, to) => {
    if (!main.includes(from)) throw Error('Canonical normalization reduction boundary not found');
    main = main.replace(from, to);
  };
  replace('var local_sum_sq: f32 = 0.0;', 'var local_sum_sq: f32 = 0.0;\n    var local_correction: f32 = 0.0;');
  replace('local_sum_sq = local_sum_sq + x * x;', `let square = x * x;
            let next_sum = local_sum_sq + square;
            local_correction += rmsnorm_sum_error(local_sum_sq, square, next_sum);
            local_sum_sq = next_sum;`);
  replace('shared_sum[thread_idx] = local_sum_sq;', `shared_sum[thread_idx] = local_sum_sq;
    rmsnorm_correction[thread_idx] = local_correction;`);
  replace('shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];',
    `let a = shared_sum[thread_idx];
            let b = shared_sum[thread_idx + stride];
            let next_sum = a + b;
            rmsnorm_correction[thread_idx] += rmsnorm_correction[thread_idx + stride]
                + rmsnorm_sum_error(a, b, next_sum);
            shared_sum[thread_idx] = next_sum;`);
  replace('let mean_sq = shared_sum[0] / f32(size);',
    'let mean_sq = (shared_sum[0] + rmsnorm_correction[0]) / f32(size);');
  return source.slice(0, start) + main + source.slice(end) + `
// Test-only compensated reduction; all operations and storage remain f32.
var<workgroup> rmsnorm_correction: array<f32, MAX_WORKGROUP_SIZE>;
fn rmsnorm_sum_error(a: f32, b: f32, sum: f32) -> f32 {
    let high = max(a, b);
    let low = min(a, b);
    return low - (sum - high);
}
`;
}

export function buildReciprocalRootDiagnostic(source, entryPoint) {
  if (!['main', 'main_subgroup'].includes(entryPoint)) throw Error('Unsupported normalization entry point');
  const boundary = entryPoint === 'main' ? 'let inv_rms = 1.0 / rms;' : 'let inv_rms = 1.0 / sqrt(mean_sq + u.eps);';
  // Avoid a separately rounded square root. Account for the squared estimate's
  // rounding with an FMA residual; every operation remains f32.
  const candidate = source.replaceAll(boundary, `let a = mean_sq + u.eps;
    let estimate = inverseSqrt(a);
    let estimate_squared = estimate * estimate;
    let square_error = fma(estimate, estimate, -estimate_squared);
    let reciprocal_residual = fma(-a, estimate_squared, 1.0) - a * square_error;
    let inv_rms = fma(0.5 * estimate, reciprocal_residual, estimate);`);
  if (candidate === source) throw Error('Canonical normalization boundary not found');
  return candidate;
}

export async function diagnoseRMSNorm({ source, input, weights, epsilon, weightOffset,
  entryPoint = 'main', weightDtype = 'f32', experiment = null }) {
  if (experiment !== null && (experiment !== 'compensated-sum' || entryPoint !== 'main')) {
    throw Error('Compensated reduction diagnostic requires the declared main entry');
  }
  if (!['main', 'main_subgroup'].includes(entryPoint) || !['f16', 'f32'].includes(weightDtype)) {
    throw Error('Explicit supported entry point and weight dtype required');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw Error('WebGPU adapter required');
  const device = await adapter.requestDevice({ requiredFeatures: ['subgroups'] });
  const size = input.length, results = [];
  const boundary = entryPoint === 'main' ? 'let inv_rms = 1.0 / rms;' : 'let inv_rms = 1.0 / sqrt(mean_sq + u.eps);';
  const candidate = source.replaceAll(boundary, `${entryPoint === 'main' ? '' : 'let rms = sqrt(mean_sq + u.eps);'}
    let a = mean_sq + u.eps;
    let root_error = fma(-rms, rms, a);
    let refined_root = rms + root_error / (2.0 * rms);
    let reciprocal = 1.0 / refined_root;
    let reciprocal_error = fma(-refined_root, reciprocal, 1.0);
    let inv_rms = fma(reciprocal, reciprocal_error, reciprocal);`);
  if (candidate === source) throw Error('Canonical normalization boundary not found');
  const reciprocalCandidate = buildReciprocalRootDiagnostic(source, entryPoint);
  const observed = source.replaceAll(boundary, `${boundary}
    if (thread_idx == 0u) {
      normalization_trace[0] = mean_sq;
      normalization_trace[1] = mean_sq + u.eps;
      normalization_trace[2] = sqrt(mean_sq + u.eps);
      normalization_trace[3] = inv_rms;
    }`) + '\n@group(0) @binding(6) var<storage, read_write> normalization_trace: array<f32>;';
  try {
    const variants = experiment === 'compensated-sum'
      ? [['canonical', source], ['compensated-sum', buildCompensatedRMSNormDiagnostic(source)]]
      : [['canonical', source], ['refined-f32-diagnostic', candidate],
        ['canonical-intermediates', observed], ['refined-rsqrt-f32-diagnostic', reciprocalCandidate]];
    for (const [variant, code] of variants) {
      const owned = [];
      const buffer = (bytes, usage) => { const value = device.createBuffer({ size: bytes, usage }); owned.push(value); return value; };
      try {
        const shader = device.createShaderModule({ code });
        const errors = (await shader.getCompilationInfo()).messages.filter(message => message.type === 'error');
        if (errors.length) throw Error(errors.map(error => error.message).join('\n'));
        const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: shader,
          entryPoint, constants: { WORKGROUP_SIZE: 256, RMS_NORM_OFFSET: weightOffset, WEIGHT_IS_F16: weightDtype === 'f16' } } });
        const uniform = buffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        const data = new ArrayBuffer(32), view = new DataView(data);
        view.setUint32(0, size, true); view.setUint32(4, 1, true); view.setFloat32(8, epsilon, true);
        view.setUint32(16, 1, true); view.setFloat32(20, 1, true); device.queue.writeBuffer(uniform, 0, data);
        const x = buffer(size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const weightData = weightDtype === 'f16' ? new Uint16Array(weights) : new Float32Array(weights);
        const w = buffer(weightData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const y = buffer(size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const residual = buffer(size * 4, GPUBufferUsage.STORAGE);
        const prenorm = buffer(size * 4, GPUBufferUsage.STORAGE);
        const readback = buffer(size * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const trace = variant === 'canonical-intermediates'
          ? buffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) : null;
        const traceReadback = trace ? buffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ) : null;
        device.queue.writeBuffer(x, 0, new Float32Array(input)); device.queue.writeBuffer(w, 0, weightData);
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
          [uniform, x, w, y, residual, prenorm, ...(trace ? [trace] : [])]
            .map((buffer, binding) => ({ binding, resource: { buffer } })) });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(y, 0, readback, 0, size * 4);
        if (trace) encoder.copyBufferToBuffer(trace, 0, traceReadback, 0, 16);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const values = Array.from(new Float32Array(readback.getMappedRange().slice(0))); readback.unmap();
        let intermediates = null;
        if (traceReadback) {
          await traceReadback.mapAsync(GPUMapMode.READ);
          const [meanSquare, epsilonSum, root, reciprocal] = new Float32Array(traceReadback.getMappedRange().slice(0));
          intermediates = { meanSquare, epsilonSum, root, reciprocal }; traceReadback.unmap();
        }
        results.push({ variant, entryPoint, weightDtype, values, intermediates });
      } finally { for (const value of owned) value.destroy(); }
    }
    return { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description }, results };
  } finally { device.destroy(); }
}
