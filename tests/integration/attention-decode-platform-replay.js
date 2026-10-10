// Replay captured decode operands against a Float64 oracle. No model acceptance.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { gunzipSync } from 'node:zlib';
import { chromium } from 'playwright';
import { float16ToFloat32 } from '../../src/converter/quantizer.js';

const [fixturePath, shaderPath, destination] = process.argv.slice(2);
assert(fixturePath && shaderPath && destination, 'Supply capture, shader and receipt paths');
const fixtureBytes = gunzipSync(await readFile(fixturePath));
const fixture = JSON.parse(fixtureBytes);
const hash = value => createHash('sha256').update(value).digest('hex');
const original = await readFile(shaderPath, 'utf8');
assert.equal(hash(original), fixture.contract.shaderSha256, 'Replay requires the captured shader');
const records = fixture.data.map(host => host.captures.captures.flatMap(capture => capture.records));
const inputs = records.map(rows => rows.find(row => row.boundary === 'inputs'));
const outputs = records.map(rows => rows.find(row => row.boundary === 'outputs'));
assert(inputs.every(row => row?.numTokens === 1 && row.plan.kv.layout === 'contiguous'));
assert.equal(inputs[0].plan.id, inputs[1].plan.id);
const tensor = (row, role) => row.tensors.find(item => item.role === role);
for (const role of ['q', 'cachedK', 'cachedV']) {
  assert.equal(tensor(inputs[0], role).data, tensor(inputs[1], role).data,
    `Local-error replay requires identical ${role} operands`);
}
const values = item => {
  const bytes = Buffer.from(item.data, 'base64');
  assert.equal(bytes.length, item.bytes);
  return Array.from({ length: item.elements }, (_, index) => item.dtype === 'f16'
    ? float16ToFloat32(bytes.readUInt16LE(index * 2)) : bytes.readFloatLE(index * 4));
};
const geometry = inputs[0], kvLen = geometry.plan.kv.length;
assert.deepEqual(fixture.uniforms, [geometry.numHeads, geometry.numKVHeads, geometry.headDim,
  kvLen, 1, geometry.scale, 1, kvLen - 1, 0, 0, 0, 0, geometry.plan.kv.pageSize, 0, 0],
  'This retained reproduction covers causal contiguous attention without softcap or a window');
const q = values(tensor(geometry, 'q'));
const k = values(tensor(geometry, 'cachedK')), v = values(tensor(geometry, 'cachedV'));
const expected = [];
for (let head = 0; head < geometry.numHeads; head++) {
  const kvHead = Math.floor(head / (geometry.numHeads / geometry.numKVHeads));
  const scores = Array.from({ length: kvLen }, (_, position) => {
    let sum = 0;
    for (let d = 0; d < geometry.headDim; d++) {
      sum += q[head * geometry.headDim + d]
        * k[(position * geometry.numKVHeads + kvHead) * geometry.headDim + d];
    }
    return sum * geometry.scale;
  });
  const maximum = Math.max(...scores), weights = scores.map(score => Math.exp(score - maximum));
  const denominator = weights.reduce((sum, weight) => sum + weight, 0);
  for (let d = 0; d < geometry.headDim; d++) {
    let sum = 0;
    for (let position = 0; position < kvLen; position++) {
      sum += weights[position] * v[(position * geometry.numKVHeads + kvHead) * geometry.headDim + d];
    }
    expected.push(sum / denominator);
  }
}
const candidate = process.env.DOPPLER_ATTENTION_DIAGNOSTIC ?? null;
const observeSoftmax = process.env.DOPPLER_ATTENTION_OBSERVE_SOFTMAX === '1';
const observeProducts = process.env.DOPPLER_ATTENTION_OBSERVE_PRODUCTS === '1';
const observeScoreLow = process.env.DOPPLER_ATTENTION_OBSERVE_SCORE_LOW === '1';
assert(!observeScoreLow || (observeSoftmax && candidate === 'compensated-attention-ratio'));
assert(!observeSoftmax || candidate === null || candidate === 'compensated-attention-ratio');
assert(!observeProducts || (candidate === null && !observeSoftmax), 'Observe one unchanged arithmetic boundary');
assert(candidate === null || (['compensated-qk', 'compensated-softmax', 'compensated-attention', 'compensated-attention-score', 'compensated-attention-ratio', 'compensated-attention-portable', 'compensated-attention-ordered'].includes(candidate)
  && process.env.DOPPLER_TEST_ONLY_ARITHMETIC === '1'), 'Arithmetic intervention requires test-only authorization');
let source = original;
if (candidate === 'compensated-qk' || (candidate?.startsWith('compensated-attention')
  && source.includes('dot = dot + q0 * k0;'))) {
  assert.equal(source.split('                var dot: f32 = 0.0;').length, 2);
  source = source.replace('                var dot: f32 = 0.0;',
    '                var dot: f32 = 0.0;\n                var correction: f32 = 0.0;');
  for (const suffix of ['0', '1']) {
    const line = `dot = dot + q${suffix} * k${suffix};`;
    assert.equal(source.split(line).length, 2);
    source = source.replace(line, `let product${suffix} = fma(q${suffix}, k${suffix}, 0.0);
                    let total${suffix} = fma(1.0, dot, product${suffix});
                    let error${suffix} = select(fma(1.0, dot, fma(-1.0, total${suffix}, product${suffix})),
                        fma(1.0, product${suffix}, fma(-1.0, total${suffix}, dot)), abs(dot) >= abs(product${suffix}));
                    correction = fma(1.0, correction, error${suffix});
                    dot = total${suffix};`);
  }
  source = source.replace('score = dot * u.scale;', 'score = fma(1.0, dot, correction) * u.scale;');
}
if (candidate === 'compensated-softmax' || candidate?.startsWith('compensated-attention')) {
  const replace = (before, after) => {
    assert.equal(source.split(before).length, 2, `Missing softmax boundary: ${before}`);
    source = source.replace(before, after);
  };
  const helpers = await readFile(new URL('../../src/gpu/kernels/silu.wgsl', import.meta.url), 'utf8');
  source += '\n' + helpers.slice(helpers.indexOf('fn exp_refined('));
  replace('exp(running_max - new_max)', 'exp_refined(running_max - new_max)');
  replace('exp(score - new_max)', 'exp_refined(score - new_max)');
  const start = source.indexOf('        var chunk_sum = subgroupAdd(exp_score);');
  const end = source.indexOf('        running_sum = running_sum * rescale + global_sum;', start);
  assert(start >= 0 && end > start);
  source = source.slice(0, start) + `        workgroupBarrier();
        if (tid == 0u) {
            var sum = 0.0;
            var correction = 0.0;
            let count = min(WORKGROUP_SIZE, kv_len - k_start);
            for (var k = 0u; k < count; k++) {
                let value = shared_scores[k];
                let total = fma(1.0, sum, value);
                let error = select(fma(1.0, sum, fma(-1.0, total, value)),
                    fma(1.0, value, fma(-1.0, total, sum)), abs(sum) >= abs(value));
                correction = fma(1.0, correction, error);
                sum = total;
            }
            global_sum = fma(1.0, sum, correction);
        }
        workgroupBarrier();

` + source.slice(end);
  replace('    var out_accum0: f32 = 0.0;',
    '    var out_accum0: f32 = 0.0;\n    var out_correction0: f32 = 0.0;');
  replace('    var out_accum1: f32 = 0.0;',
    '    var out_accum1: f32 = 0.0;\n    var out_correction1: f32 = 0.0;');
  for (const suffix of ['0', '1']) {
    replace(`            out_accum${suffix} = out_accum${suffix} * rescale;`,
      `            out_accum${suffix} = fma(out_accum${suffix}, rescale, 0.0);
            out_correction${suffix} = fma(out_correction${suffix}, rescale, 0.0);`);
    replace(`                    out_accum${suffix} = out_accum${suffix} + shared_scores[score_idx] * f32(V[v_base + out_dim${suffix}]);`,
      `                    let value = f32(V[v_base + out_dim${suffix}]);
                    let probability = shared_scores[score_idx];
                    let product = fma(probability, value, 0.0);
                    let total = fma(1.0, out_accum${suffix}, product);
                    let error = select(fma(1.0, out_accum${suffix}, fma(-1.0, total, product)),
                        fma(1.0, product, fma(-1.0, total, out_accum${suffix})), abs(out_accum${suffix}) >= abs(product));
                    out_correction${suffix} = fma(1.0, out_correction${suffix}, fma(probability, value, -product) + error);
                    out_accum${suffix} = total;`);
    replace(`out_accum${suffix} * inv_sum;`, `fma(1.0, out_accum${suffix}, out_correction${suffix}) * inv_sum;`);
  }
  replace('1.0 / running_sum, running_sum > 0.0', 'reciprocal_refined(running_sum), running_sum > 0.0');
  if (candidate?.startsWith('compensated-attention')) {
    for (const suffix of ['0', '1']) {
      replace(`correction = fma(1.0, correction, error${suffix});`,
        `correction = fma(1.0, correction, fma(1.0, fma(q${suffix}, k${suffix}, -product${suffix}), error${suffix}));`);
    }
  }
  if (['compensated-attention-score', 'compensated-attention-ratio', 'compensated-attention-portable', 'compensated-attention-ordered'].includes(candidate)) {
    replace('        var score: f32 = -3.402823e+38;',
      '        var score: f32 = -3.402823e+38;\n        var score_correction: f32 = 0.0;');
    replace('                score = fma(1.0, dot, correction) * u.scale;',
      `                let dot_sum = fma(1.0, dot, correction);
                let dot_error = select(fma(1.0, dot, fma(-1.0, dot_sum, correction)),
                    fma(1.0, correction, fma(-1.0, dot_sum, dot)), abs(dot) >= abs(correction));
                score = dot_sum * u.scale;
                score_correction = fma(dot_sum, u.scale, -score) + dot_error * u.scale;`);
    replace('                    score = tanh(score / u.attn_softcap) * u.attn_softcap;',
      '                    score = tanh(score / u.attn_softcap) * u.attn_softcap;\n                    score_correction = 0.0;');
    replace('            exp_score = exp_refined(score - new_max);',
      `            let difference = fma(-1.0, new_max, score);
            let error = select(fma(1.0, score, fma(-1.0, difference, -new_max)),
                fma(-1.0, new_max, fma(-1.0, difference, score)), abs(score) >= abs(new_max));
            let low = fma(1.0, score_correction, error);
            let exponential = exp_refined(difference);
            exp_score = fma(exponential, low, exponential);`);
  }
  if (['compensated-attention-ratio', 'compensated-attention-portable', 'compensated-attention-ordered'].includes(candidate)) {
    replace('var<workgroup> global_sum: f32;',
      'var<workgroup> global_sum: f32;\nvar<workgroup> global_sum_correction: f32;');
    replace('    var running_sum: f32 = 0.0;',
      '    var running_sum: f32 = 0.0;\n    var running_correction: f32 = 0.0;');
    replace('            global_sum = fma(1.0, sum, correction);',
      '            global_sum = sum;\n            global_sum_correction = correction;');
    replace('        running_sum = running_sum * rescale + global_sum;',
      `        let rescaled = fma(running_sum, rescale, 0.0);
        let rescale_error = fma(running_sum, rescale, -rescaled) + running_correction * rescale;
        let total_sum = fma(1.0, rescaled, global_sum);
        let sum_error = select(fma(1.0, rescaled, fma(-1.0, total_sum, global_sum)),
            fma(1.0, global_sum, fma(-1.0, total_sum, rescaled)), abs(rescaled) >= abs(global_sum));
        running_correction = fma(1.0, rescale_error, fma(1.0, global_sum_correction, sum_error));
        running_sum = total_sum;`);
    for (const suffix of ['0', '1']) {
      replace(`output[q_offset + out_dim${suffix}] = fma(1.0, out_accum${suffix}, out_correction${suffix}) * inv_sum;`,
        `let quotient = out_accum${suffix} * inv_sum;
        let residual = fma(-quotient, running_sum, out_accum${suffix})
            + fma(-quotient, running_correction, out_correction${suffix});
        output[q_offset + out_dim${suffix}] = fma(residual, inv_sum, quotient);`);
    }
  }
  if (candidate === 'compensated-attention-portable') {
    source += `
// F32 times stored F16: both split products fit the F32 significand exactly.
fn product_error_f16(a: f32, b: f32, rounded: f32) -> f32 {
    let high = bitcast<f32>(bitcast<u32>(a) & 0xfffff000u);
    let low = fma(-1.0, high, a);
    return fma(1.0, fma(high, b, -rounded), fma(low, b, 0.0));
}
`;
    for (const suffix of ['0', '1']) replace(`fma(q${suffix}, k${suffix}, -product${suffix})`,
      `product_error_f16(q${suffix}, k${suffix}, product${suffix})`);
    assert.equal(source.split('fma(probability, value, -product)').length, 3);
    source = source.replaceAll('fma(probability, value, -product)', 'product_error_f16(probability, value, product)');
  }
  if (candidate === 'compensated-attention-ordered') {
    source += `
fn sum_ordered(a: f32, b: f32, c: f32) -> f32 {
    let terms = array<f32, 3>(a, b, c);
    var sum = 0.0;
    for (var i = 0u; i < 3u; i++) { sum = fma(1.0, sum, terms[i]); }
    return sum;
}
`;
    replace(`let dot_error = select(fma(1.0, dot, fma(-1.0, dot_sum, correction)),
                    fma(1.0, correction, fma(-1.0, dot_sum, dot)), abs(dot) >= abs(correction));`,
      `let dot_error = select(sum_ordered(correction, -dot_sum, dot),
                    sum_ordered(dot, -dot_sum, correction), abs(dot) >= abs(correction));`);
    replace(`let error = select(fma(1.0, score, fma(-1.0, difference, -new_max)),
                fma(-1.0, new_max, fma(-1.0, difference, score)), abs(score) >= abs(new_max));`,
      `let error = select(sum_ordered(-new_max, -difference, score),
                sum_ordered(score, -difference, -new_max), abs(score) >= abs(new_max));`);
  }
}
if (observeSoftmax) {
  assert(kvLen <= 256, 'This observation covers one online softmax chunk');
  source += '\n@group(0) @binding(7) var<storage, read_write> observed_softmax: array<f32>;\n';
  source = source.replace('        shared_scores[tid] = exp_score;', `        shared_scores[tid] = exp_score;
        let observed_base = head_idx * (2u * kv_len + 4u + head_dim);
        if (tid < kv_len) {
            observed_softmax[observed_base + tid] = score;
            observed_softmax[observed_base + kv_len + tid] = exp_score;
            ${observeScoreLow ? 'observed_softmax[observed_base + 2u * kv_len + 4u + tid] = score_correction;' : ''}
        }`);
  const invLine = candidate === 'compensated-attention-ratio'
    ? '    let inv_sum = select(0.0, reciprocal_refined(running_sum), running_sum > 0.0);'
    : '    let inv_sum = select(0.0, 1.0 / running_sum, running_sum > 0.0);';
  assert.equal(source.split(invLine).length, 2);
  source = source.replace(invLine,
    `${invLine}
    let observed_base = head_idx * (2u * kv_len + 4u + head_dim);
    if (tid == 0u) {
        observed_softmax[observed_base + 2u * kv_len] = running_sum;
        observed_softmax[observed_base + 2u * kv_len + 1u] = inv_sum;
        observed_softmax[observed_base + 2u * kv_len + 2u] = running_max;
        observed_softmax[observed_base + 2u * kv_len + 3u] = f32(subgroup_size);
    }
    ${observeScoreLow ? '' : `if (has_out_dim0) { observed_softmax[observed_base + 2u * kv_len + 4u + out_dim0] = out_accum0; }
    if (has_out_dim1) { observed_softmax[observed_base + 2u * kv_len + 4u + out_dim1] = out_accum1; }`}`);
}
if (observeProducts) {
  assert(kvLen <= 256);
  source += '\n@group(0) @binding(7) var<storage, read_write> observed_products: array<vec2<f32>>;\n';
  for (const suffix of ['0', '1']) {
    const before = `let product${suffix} = fma(q${suffix}, k${suffix}, 0.0);`;
    assert.equal(source.split(before).length, 2);
    source = source.replace(before, `${before}
                    if (d < 32u) {
                        observed_products[(head_idx * kv_len + k_pos) * 32u + d + ${suffix}u] =
                            vec2<f32>(product${suffix}, fma(q${suffix}, k${suffix}, -product${suffix}));
                    }`);
  }
}
const backends = { darwin: ['--use-angle=metal'],
  linux: ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'] };
assert(Object.hasOwn(backends, process.platform));
const host = process.platform === 'darwin' ? 'mac' : 'linux';
const captured = values(tensor(outputs[fixture.data.findIndex(row => row.host === host)], 'core'));
const receipt = { scope: 'Identical captured operands; isolated decode attention, not model acceptance',
  host, candidate, observeSoftmax, observeProducts, observeScoreLow, sourceSubstitution: candidate !== null || observeSoftmax || observeProducts, fixtureSha256: hash(fixtureBytes),
  originalShaderSha256: hash(original), shaderSha256: hash(source), contract: fixture.contract };
const compare = (actual, reference) => {
  assert.equal(actual.length, reference.length);
  const errors = actual.map((value, index) => Math.abs(value - reference[index]));
  assert(errors.every(Number.isFinite));
  return { maxError: Math.max(...errors), rmsError: Math.sqrt(errors.reduce((s, e) => s + e * e, 0) / errors.length) };
};
const server = createServer((_request, response) => response.end('<!doctype html>'));
await new Promise(done => server.listen(0, '127.0.0.1', done));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true,
    args: ['--enable-unsafe-webgpu', ...backends[process.platform]] });
  receipt.browser = browser.version();
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result = await page.evaluate(async ({ source, fixture, geometry, observeSoftmax, observeProducts }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw Error('Physical GPU required');
    const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16', 'subgroups'] });
    const owned = [], errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const buffer = (size, usage, data) => {
      const resource = device.createBuffer({ size, usage }); owned.push(resource);
      if (data) device.queue.writeBuffer(resource, 0, data);
      return resource;
    };
    let staging, softmaxStaging;
    try {
      const module = device.createShaderModule({ code: source });
      const diagnostics = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (diagnostics.length) throw Error(diagnostics.map(row => row.message).join('; '));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: {
        module, entryPoint: fixture.contract.entryPoint, constants: fixture.contract.constants } });
      const bytes = new ArrayBuffer(64), view = new DataView(bytes);
      const u = fixture.uniforms;
      for (const [index, value] of u.entries()) {
        if (index === 5 || index === 8) view.setFloat32(index * 4, value, true);
        else view.setUint32(index * 4, value, true);
      }
      const input = role => {
        const tensor = geometry.tensors.find(item => item.role === role);
        return buffer(tensor.bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          Uint8Array.from(atob(tensor.data), char => char.charCodeAt(0)));
      };
      const output = buffer(geometry.numHeads * geometry.headDim * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const resources = [buffer(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, bytes),
        input('q'), input('cachedK'), input('cachedV'), output,
        buffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, new Uint32Array([u[3]])),
        buffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, new Uint32Array(1))];
      const observationBytes = observeSoftmax ? geometry.numHeads * (2 * u[3] + 4 + geometry.headDim) * 4
        : observeProducts ? geometry.numHeads * u[3] * 32 * 8 : 0;
      const softmax = observationBytes ? buffer(observationBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) : null;
      if (softmax) resources.push(softmax);
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: resources.map((resource, binding) => ({ binding, resource: { buffer: resource } })) });
      staging = buffer(output.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(geometry.numHeads); pass.end();
      encoder.copyBufferToBuffer(output, 0, staging, 0, output.size);
      if (softmax) {
        softmaxStaging = buffer(softmax.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        encoder.copyBufferToBuffer(softmax, 0, softmaxStaging, 0, softmax.size);
      }
      device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
      const values = Array.from(new Float32Array(staging.getMappedRange().slice(0)));
      let softmaxValues = null;
      if (softmaxStaging) {
        await softmaxStaging.mapAsync(GPUMapMode.READ);
        softmaxValues = Array.from(new Float32Array(softmaxStaging.getMappedRange().slice(0)));
      }
      return { values, softmaxValues, errors, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } };
    } finally {
      if (staging?.mapState === 'mapped') staging.unmap();
      if (softmaxStaging?.mapState === 'mapped') softmaxStaging.unmap();
      await device.queue.onSubmittedWorkDone().catch(() => {});
      for (const resource of owned) resource.destroy(); device.destroy();
    }
  }, { source, fixture, geometry, observeSoftmax, observeProducts });
  Object.assign(receipt, result);
  receipt.comparison = compare(result.values, expected);
  receipt.capturedReference = compare(captured, expected);
  receipt.capturedReplay = compare(result.values, captured);
  receipt.valuesSha256 = hash(Buffer.from(Float32Array.from(result.values).buffer));
  if (observeProducts) {
    const actual = result.softmaxValues;
    let mismatchedProducts = 0, nonzeroResiduals = 0, missingResiduals = 0, maxResidualError = 0;
    for (let head = 0; head < geometry.numHeads; head++) {
      const kvHead = Math.floor(head / (geometry.numHeads / geometry.numKVHeads));
      for (let position = 0; position < kvLen; position++) for (let d = 0; d < 32; d++) {
        const product = q[head * geometry.headDim + d] * k[(position * geometry.numKVHeads + kvHead) * geometry.headDim + d];
        const rounded = Math.fround(product), residual = Math.fround(product - rounded);
        const offset = ((head * kvLen + position) * 32 + d) * 2;
        if (rounded !== actual[offset]) mismatchedProducts++;
        if (residual !== 0) {
          nonzeroResiduals++;
          if (actual[offset + 1] === 0) missingResiduals++;
        }
        maxResidualError = Math.max(maxResidualError, Math.abs(residual - actual[offset + 1]));
      }
    }
    receipt.productResidual = { scope: 'First 32 dimensions of captured F32 Q and F16 K; exact Float64 products',
      products: geometry.numHeads * kvLen * 32, mismatchedProducts, nonzeroResiduals, missingResiduals, maxResidualError };
  }
  assert.deepEqual(result.errors, []);
  if (!candidate) assert.equal(receipt.capturedReplay.maxError, 0, 'Baseline must reproduce actual captured output');
  else {
    receipt.regressionMaxError = 4e-7;
    assert(receipt.comparison.maxError <= receipt.regressionMaxError,
      'Captured decode correction exceeds the independent regression bound');
    assert(receipt.comparison.maxError < receipt.capturedReference.maxError / 2,
      'Correction must materially improve independent accuracy');
  }
  receipt.passed = true;
  console.log(JSON.stringify({ host, candidate, comparison: receipt.comparison, capturedReplay: receipt.capturedReplay,
    valuesSha256: receipt.valuesSha256 }));
} catch (error) { receipt.passed = false; receipt.failure = error.message; throw error; }
finally {
  await writeFile(destination, JSON.stringify(receipt));
  await browser?.close(); await new Promise(done => server.close(done));
}
