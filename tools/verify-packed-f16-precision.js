#!/usr/bin/env node
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [runtimeRoot, manifestDir, outputFile, expectedVendor] = process.argv.slice(2);
if (!runtimeRoot || !manifestDir || !outputFile || !expectedVendor) {
  console.error('Usage: node verify-packed-f16-precision.js <unpacked-dopplerx-dir> <manifest-dir> <output.json> <vendor-regex>');
  process.exit(2);
}

const load = (relPath) => import(pathToFileURL(path.resolve(runtimeRoot, relPath)).href);
const pkg = JSON.parse(readFileSync(path.resolve(runtimeRoot, 'package.json'), 'utf8'));
assert.equal(pkg.name, 'dopplerx', `Expected package name dopplerx, got ${pkg.name}`);

const { bootstrapNodeWebGPU } = await load('src/tooling/node-webgpu.js');
const gpuModule = await load('src/gpu/device.js');
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

try {
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

  assert.deepEqual(errors, []);
  const receipt = {
    schema: 'dopplerx.physical-f16-qualification/v1',
    package: { name: pkg.name, version: pkg.version },
    device: hw,
    features: [...device.features].sort(),
    models,
    failClosedChecks,
    checkedUtc: new Date().toISOString(),
  };
  writeFileSync(outputFile, JSON.stringify(receipt, null, 2) + '\n');
  console.log(`Physical F16 & fail-closed check passed on ${hw.vendor} (${hw.device || hw.architecture || 'gpu'}) for ${models.length} manifests.`);
} finally {
  await gpuModule.destroyDevice();
}
