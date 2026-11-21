#!/usr/bin/env node
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [runtimeRoot, manifestDir, outputFile, expectedVendor, tarballPath] = process.argv.slice(2);
if (!runtimeRoot || !manifestDir || !outputFile || !expectedVendor) {
  console.error('Usage: node verify-packed-f16-precision.js <unpacked-dopplerx-dir> <manifest-dir> <output.json> <vendor-regex> [tarball-path]');
  process.exit(2);
}

function countFiles(dir) {
  let count = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === 'node_modules') continue;
    const full = path.join(dir, entry.name);
    if (entry.isSymbolicLink()) continue;
    if (entry.isDirectory()) count += countFiles(full);
    else if (statSync(full).isFile()) count += 1;
  }
  return count;
}

const archiveSha256 = tarballPath
  ? createHash('sha256').update(readFileSync(tarballPath)).digest('hex')
  : null;
const unpackedFileCount = countFiles(runtimeRoot);

const load = (relPath) => import(pathToFileURL(path.resolve(runtimeRoot, relPath)).href);
const pkg = JSON.parse(readFileSync(path.resolve(runtimeRoot, 'package.json'), 'utf8'));
assert.equal(pkg.name, 'dopplerx', `Expected package name dopplerx, got ${pkg.name}`);

const { bootstrapNodeWebGPU } = await load('src/tooling/node-webgpu.js');
const gpuModule = await load('src/gpu/device.js');
const { acquireBuffer, releaseBuffer, readBuffer, uploadData, getBufferPool } = await load('src/memory/buffer-pool.js');
const { createTensor } = await load('src/gpu/tensor.js');
const { clipGradients } = await load('src/experimental/training/clip.js');
const { detectOverflow } = await load('src/experimental/training/loss-scaling.js');
const { applyExecutionV1RuntimeConfig } = await load('src/inference/pipelines/text/execution/execution-v1.js');
const { resolveKernelPathState } = await load('src/inference/pipelines/text/loading/model-load.js');
const { getLayerSteps } = await load('src/config/kernel-path-loader.js');

const bootstrap = await bootstrapNodeWebGPU();
assert.equal(bootstrap.ok, true, `WebGPU bootstrap failed: ${bootstrap.detail || 'unknown'}`);
await gpuModule.initDevice();
const device = gpuModule.getDevice();
const caps = gpuModule.getKernelCapabilities();
const hw = caps.adapterInfo;
assert(hw && hw.vendor, 'Missing physical WebGPU adapter vendor');
assert(new RegExp(expectedVendor, 'i').test(JSON.stringify(hw)), `Unexpected adapter ${JSON.stringify(hw)}`);
assert(!/swiftshader|llvmpipe|lavapipe|software/i.test(JSON.stringify(hw)), `Software adapter rejected: ${JSON.stringify(hw)}`);
assert(caps.hasF16 && caps.hasSubgroups, 'Physical shader-f16 and subgroups required');

const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const B = GPUBufferUsage;
const models = [];
const failClosedChecks = [];
let gradientClippingAndOverflow = null;

try {
  // 1. Verify packed F16 Qwen kernel dispatches and fail-closed precision rejection across all four manifests.
  for (const file of ['0-8b-old.json', '0-8b.json', '2b-old.json', '2b.json']) {
    const manifestPath = path.join(manifestDir, file);
    const manifestBytes = readFileSync(manifestPath);
    const manifest = JSON.parse(manifestBytes.toString('utf8'));
    const f16Conf = {
      inference: {
        compute: { activationDtype: 'f16' },
        kernelPathPolicy: { mode: 'capability-aware', onIncompatible: 'remap', sourceScope: ['manifest'] },
      },
    };
    const compiled = applyExecutionV1RuntimeConfig({
      manifest,
      runtimeConfig: f16Conf,
      runtimeOverrides: f16Conf,
      capabilities: caps,
    });
    const state = compiled.executionV1State;
    assert(
      state.appliedTransforms.includes('useQwenF16PrimaryMatmuls'),
      `Expected useQwenF16PrimaryMatmuls for ${manifest.modelId}, got ${JSON.stringify(state.appliedTransforms)}`
    );
    resolveKernelPathState({
      manifest,
      runtimeConfig: compiled.runtimeConfig,
      runtimeOverrides: f16Conf,
      executionV1State: state,
      modelConfig: { kernelPath: null },
      kernelCapabilities: caps,
    });
    const kernelPath = state.runtimeInferencePatch.kernelPath;
    const layer = manifest.inference.layerPattern.layerTypes.findIndex((x) => /full/i.test(x));
    assert(layer >= 0);
    const step = getLayerSteps(kernelPath, layer, 'decode').find((x) => x.op === 'q_proj');
    assert.deepEqual(step.precision, { inputDtype: 'f16', outputDtype: 'f16' });
    assert.equal(step.kernel, 'fused_matmul_q4_multicol_f16a.wgsl');

    const code = readFileSync(path.resolve(runtimeRoot, 'src/gpu/kernels', step.kernel), 'utf8');
    const shader = device.createShaderModule({ code });
    const diagnostics = await shader.getCompilationInfo();
    assert(!diagnostics.messages.some((x) => x.type === 'error'), diagnostics.messages.map((x) => x.message).join('\n'));
    device.pushErrorScope('validation');
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: shader, entryPoint: step.entry, constants: step.constants || {} },
    });

    const allocated = [];
    const alloc = (size, usage) => {
      const b = device.createBuffer({ size, usage });
      allocated.push(b);
      return b;
    };
    try {
      const parameters = new Uint32Array([1, 2, 256, 0x3f800000, 1, 0, 0, 0]);
      const weights = new Uint8Array(288);
      for (let row = 0; row < 2; row += 1) {
        const offset = row * 144;
        weights.set([0, 60, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1], offset);
        weights.set([0, 60, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], offset);
        weights.fill(row === 0 ? 0x11 : 0x22, offset + 16, offset + 144);
      }
      const data = [parameters, new Uint16Array(256).fill(0x3c00), weights, new Uint16Array([0xffff, 0xffff])];
      const bindings = [0, 1, 2, 4];
      const bindEntries = data.map((arr, idx) => {
        const usage = (idx === 0 ? B.UNIFORM : B.STORAGE) | B.COPY_DST | (idx === 3 ? B.COPY_SRC : 0);
        const buffer = alloc(arr.byteLength, usage);
        device.queue.writeBuffer(buffer, 0, arr);
        return { binding: bindings[idx], resource: { buffer } };
      });
      const readback = alloc(4, B.COPY_DST | B.MAP_READ);
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: bindEntries });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(1);
      pass.end();
      enc.copyBufferToBuffer(bindEntries[3].resource.buffer, 0, readback, 0, 4);
      device.queue.submit([enc.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = Array.from(new Uint16Array(readback.getMappedRange()));
      readback.unmap();
      assert.deepEqual(actual, [0x5c00, 0x6000], `Expected exact F16 bits [0x5c00, 0x6000] (256, 512), got ${JSON.stringify(actual)}`);
      const validation = await device.popErrorScope();
      assert.equal(validation, null);
      models.push({
        modelId: manifest.modelId,
        manifestSha256: createHash('sha256').update(manifestBytes).digest('hex'),
        layer,
        kernel: step.kernel,
        entry: step.entry,
        shaderSha256: createHash('sha256').update(code).digest('hex'),
        outputF16Bits: actual,
      });
    } finally {
      for (const b of allocated) b.destroy();
    }

    // Fail-closed check: explicit F16 request when device lacks shader-f16 must throw before any GPU dispatch.
    const noF16Caps = { ...caps, hasF16: false, hasSubgroupsF16: false };
    let rejectedMessage = null;
    try {
      const strictConf = {
        inference: {
          compute: { activationDtype: 'f16' },
          kernelPathPolicy: { mode: 'capability-aware', onIncompatible: 'error', sourceScope: ['manifest'] },
        },
      };
      const attempt = applyExecutionV1RuntimeConfig({
        manifest,
        runtimeConfig: strictConf,
        runtimeOverrides: strictConf,
        capabilities: noF16Caps,
      });
      resolveKernelPathState({
        manifest,
        runtimeConfig: attempt.runtimeConfig,
        runtimeOverrides: strictConf,
        executionV1State: attempt.executionV1State,
        modelConfig: { kernelPath: null },
        kernelCapabilities: noF16Caps,
      });
    } catch (err) {
      rejectedMessage = err?.message || String(err);
    }
    assert(rejectedMessage, `Expected fail-closed rejection when shader-f16 is unavailable for ${manifest.modelId}`);
    failClosedChecks.push({
      modelId: manifest.modelId,
      rejectedBeforeDispatch: true,
      errorMessage: rejectedMessage,
    });
  }

  // 2. Physical GPU gradient clipping & overflow check with leftover NaN/Inf/200 data beyond logical tensor extents.
  getBufferPool().configure({ enablePooling: true });
  {
    // F32 pooled buffer: 16 bytes physical ([3, 4, NaN, 200]), logical shape [2] (8 bytes).
    const f32Backing = new Float32Array([3, 4, Number.NaN, 200]);
    const f32Buf = acquireBuffer(f32Backing.byteLength, B.STORAGE | B.COPY_DST | B.COPY_SRC, 'grad_f32_padded');
    uploadData(f32Buf, f32Backing);
    const f32Grad = createTensor(f32Buf, 'f32', [2], 'grad_f32_padded');
    const f32Map = new Map([['weight', f32Grad]]);

    const f32OverflowClean = await detectOverflow(f32Map);
    assert.equal(f32OverflowClean, false, 'detectOverflow(f32) must ignore trailing NaN in pooled buffer padding');

    const f32ClipRes = await clipGradients(f32Map, { training: { gradientClipping: { maxNorm: 1 } } });
    assert.equal(f32ClipRes.total_param_count, 2);
    assert.equal(f32ClipRes.clipped_event_count, 1);
    assert.equal(f32ClipRes.gradient_norm_unclipped, 5);
    assert.equal(f32ClipRes.gradient_norm_clipped, 1);
    const f32ClippedTensor = f32ClipRes.clippedGrads.get('weight');
    const f32ClippedValues = Array.from(new Float32Array(await readBuffer(f32ClippedTensor.buffer, 8))).map((v) => Number(v.toFixed(6)));
    assert.deepEqual(f32ClippedValues, [0.6, 0.8], `Expected physical GPU scaled gradients [0.6, 0.8], got ${JSON.stringify(f32ClippedValues)}`);

    // Inject NaN inside logical extent [0..1] -> detectOverflow must return true.
    uploadData(f32Buf, new Float32Array([Number.NaN, 4, 0, 0]));
    const f32OverflowDetected = await detectOverflow(f32Map);
    assert.equal(f32OverflowDetected, true, 'detectOverflow(f32) must detect NaN within logical extent');
    releaseBuffer(f32ClippedTensor.buffer);
    releaseBuffer(f32Buf);

    // F16 pooled buffer: 16 bytes physical ([3.0, 4.0, NaN, +Inf, 200.0, ...]), logical shape [2] (4 bytes).
    // IEEE-754 F16: 3.0 = 0x4200, 4.0 = 0x4400, NaN = 0x7e00, +Inf = 0x7c00
    const f16Backing = new Uint16Array([0x4200, 0x4400, 0x7e00, 0x7c00, 0x7e00, 0x7c00, 0x7e00, 0x7c00]);
    const f16Buf = acquireBuffer(f16Backing.byteLength, B.STORAGE | B.COPY_DST | B.COPY_SRC, 'grad_f16_padded');
    uploadData(f16Buf, f16Backing);
    const f16Grad = createTensor(f16Buf, 'f16', [2], 'grad_f16_padded');
    const f16Map = new Map([['weight_f16', f16Grad]]);

    const f16OverflowClean = await detectOverflow(f16Map);
    assert.equal(f16OverflowClean, false, 'detectOverflow(f16) must ignore trailing NaN/+Inf in pooled buffer padding');

    const f16ClipRes = await clipGradients(f16Map, { training: { gradientClipping: { maxNorm: 1 } } });
    assert.equal(f16ClipRes.total_param_count, 2);
    assert.equal(f16ClipRes.clipped_event_count, 1);
    assert.equal(f16ClipRes.gradient_norm_unclipped, 5);
    assert.equal(f16ClipRes.gradient_norm_clipped, 1);
    const f16ClippedTensor = f16ClipRes.clippedGrads.get('weight_f16');
    const f16ClippedBits = Array.from(new Uint16Array(await readBuffer(f16ClippedTensor.buffer, 4)));
    // 0.6 in F16 is 0x38cd (0.60009765625), 0.8 in F16 is 0x3a66 (0.7998046875)
    assert.deepEqual(f16ClippedBits, [0x38cd, 0x3a66], `Expected F16 clipped bits [0x38cd, 0x3a66], got ${JSON.stringify(f16ClippedBits)}`);

    uploadData(f16Buf, new Uint16Array([0x7e00, 0x4400, 0, 0, 0, 0, 0, 0]));
    const f16OverflowDetected = await detectOverflow(f16Map);
    assert.equal(f16OverflowDetected, true, 'detectOverflow(f16) must detect NaN within logical extent');
    releaseBuffer(f16ClippedTensor.buffer);
    releaseBuffer(f16Buf);

    gradientClippingAndOverflow = {
      f32LogicalExtentCheck: {
        physicalElements: 4,
        logicalShape: [2],
        paddedInputWithTrailingNaNAnd200: [3, 4, 'NaN', 200],
        overflowWithDirtyPadding: f32OverflowClean,
        totalParamCount: f32ClipRes.total_param_count,
        unclippedNorm: f32ClipRes.gradient_norm_unclipped,
        clippedNorm: f32ClipRes.gradient_norm_clipped,
        clippedGpuValues: f32ClippedValues,
        overflowWhenLogicalElementNaN: f32OverflowDetected,
      },
      f16LogicalExtentCheck: {
        physicalElements: 8,
        logicalShape: [2],
        paddedInputF16Hex: ['0x4200', '0x4400', '0x7e00', '0x7c00'],
        overflowWithDirtyPadding: f16OverflowClean,
        totalParamCount: f16ClipRes.total_param_count,
        unclippedNorm: f16ClipRes.gradient_norm_unclipped,
        clippedNorm: f16ClipRes.gradient_norm_clipped,
        clippedGpuF16Bits: f16ClippedBits,
        overflowWhenLogicalElementNaN: f16OverflowDetected,
      },
    };
  }

  assert.deepEqual(errors, []);
  const receipt = {
    schema: 'dopplerx.physical-f16-qualification/v2',
    package: {
      name: pkg.name,
      version: pkg.version,
      archiveSha256,
      unpackedFileCount,
    },
    device: hw,
    features: [...device.features].sort(),
    gradientClippingAndOverflow,
    models,
    failClosedChecks,
    checkedUtc: new Date().toISOString(),
  };
  writeFileSync(outputFile, JSON.stringify(receipt, null, 2) + '\n');
  console.log(JSON.stringify({
    archiveSha256,
    unpackedFileCount,
    vendor: hw.vendor,
    f32Clipped: gradientClippingAndOverflow.f32LogicalExtentCheck.clippedGpuValues,
    f16ClippedBits: gradientClippingAndOverflow.f16LogicalExtentCheck.clippedGpuF16Bits,
    modelsVerified: models.length,
  }));
} finally {
  device.destroy();
}
process.exit(0);
