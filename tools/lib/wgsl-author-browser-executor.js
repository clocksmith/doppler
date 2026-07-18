import {
  hashWgslSemanticEvidenceValue,
} from '../../src/tooling/wgsl-repair-semantic-gate.js';
import { settleWithTimeout } from './wgsl-browser-verifier.js';

const DEFAULT_BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-dawn-features=allow_unsafe_apis',
  '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE',
  '--use-angle=vulkan',
  '--disable-vulkan-surface',
]);

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function requirePositiveInteger(value, fallback, label) {
  const parsed = value == null ? fallback : Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive safe integer.`);
  }
  return parsed;
}

function normalizeRequiredLimits(value) {
  if (!isPlainObject(value)) throw new Error('requiredLimits must be an object.');
  return Object.fromEntries(Object.entries(value).map(([name, minimum]) => {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/u.test(name)
      || !Number.isSafeInteger(minimum)
      || minimum < 0) {
      throw new Error(`Invalid required WebGPU limit: ${name}.`);
    }
    return [name, minimum];
  }));
}

function normalizeCompilationMessages(messages) {
  return (messages || []).map((message) => ({
    type: String(message.type || ''),
    message: String(message.message || ''),
    lineNum: Number(message.lineNum || 0),
    linePos: Number(message.linePos || 0),
    offset: Number(message.offset || 0),
    length: Number(message.length || 0),
  }));
}

function summarizeExecution(id, plan, deviceInfo, browserArgs, result) {
  const compilation = (result.compilation || []).map((entry) => {
    const messages = normalizeCompilationMessages(entry.messages);
    const errorCount = messages.filter((message) => message.type === 'error').length;
    return {
      moduleId: entry.moduleId,
      passed: errorCount === 0,
      errorCount,
      messages,
    };
  });
  const runtimeErrors = (result.runtimeErrors || []).map(String);
  const expectedPassIds = plan.passes.map((pass) => pass.id);
  const executedPassIds = (result.executedPassIds || []).map(String);
  const outputIds = Object.keys(result.outputs || {}).sort();
  const passed = compilation.length === plan.modules.length
    && compilation.every((entry) => entry.passed)
    && runtimeErrors.length === 0
    && JSON.stringify(executedPassIds) === JSON.stringify(expectedPassIds)
    && plan.outputs.every((outputId) => outputIds.includes(outputId));
  const core = {
    schema: 'doppler.wgsl-author-browser-execution/v1',
    id,
    planSha256: hashWgslSemanticEvidenceValue(plan),
    deviceInfo,
    browserArgs,
    compilation,
    executedPassIds,
    outputs: result.outputs || {},
    runtimeErrors,
    validationErrorsAbsent: runtimeErrors.length === 0,
    passed,
  };
  return { ...core, receiptHash: hashWgslSemanticEvidenceValue(core) };
}

export async function createWgslAuthorBrowserExecutor(options = {}) {
  const { chromium } = await import('playwright');
  const browserArgs = Array.isArray(options.browserArgs)
    ? options.browserArgs.map(String)
    : [...DEFAULT_BROWSER_ARGS];
  const requiredFeatures = Array.isArray(options.requiredFeatures)
    ? [...new Set(options.requiredFeatures.map(String))].sort()
    : [];
  const requiredLimits = normalizeRequiredLimits(options.requiredLimits || {});
  const executionTimeoutMs = requirePositiveInteger(
    options.executionTimeoutMs,
    30000,
    'executionTimeoutMs'
  );
  const powerPreference = String(options.powerPreference || 'high-performance');
  const requiredVendor = String(options.requiredVendor || '').trim().toLowerCase();
  let browser = null;
  let page = null;

  async function openSession() {
    browser = await chromium.launch({
      headless: options.headless !== false,
      args: browserArgs,
    });
    page = await browser.newPage();
    await page.route('https://wgsl-author.invalid/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/html',
        body: '<!doctype html><meta charset="utf-8"><title>WGSL author executor</title>',
      });
    });
    await page.goto('https://wgsl-author.invalid/');
    const info = await page.evaluate(async (request) => {
      if (!navigator.gpu) throw new Error('webgpu_unavailable');
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: request.powerPreference,
      });
      if (!adapter) throw new Error('webgpu_adapter_unavailable');
      const missingFeatures = request.requiredFeatures.filter((feature) => (
        !adapter.features.has(feature)
      ));
      if (missingFeatures.length > 0) {
        throw new Error(`webgpu_required_features_missing:${missingFeatures.join(',')}`);
      }
      const unavailableLimits = Object.entries(request.requiredLimits).filter(([name, minimum]) => (
        typeof adapter.limits[name] !== 'number' || adapter.limits[name] < minimum
      ));
      if (unavailableLimits.length > 0) {
        throw new Error(`webgpu_required_limits_missing:${unavailableLimits
          .map(([name, minimum]) => `${name}:${minimum}`)
          .join(',')}`);
      }
      const device = await adapter.requestDevice({
        requiredFeatures: request.requiredFeatures,
        requiredLimits: request.requiredLimits,
      });
      globalThis.__wgslAuthorDevice = device;
      const adapterInfo = adapter.info || {};
      return {
        vendor: adapterInfo.vendor || null,
        architecture: adapterInfo.architecture || null,
        device: adapterInfo.device || null,
        description: adapterInfo.description || null,
        availableFeatures: [...adapter.features].sort(),
        requiredFeatures: request.requiredFeatures,
        limits: Object.fromEntries(Object.keys(request.requiredLimits).map((name) => [
          name,
          adapter.limits[name],
        ])),
        requiredLimits: request.requiredLimits,
      };
    }, { powerPreference, requiredFeatures, requiredLimits });
    if (requiredVendor && String(info.vendor || '').toLowerCase() !== requiredVendor) {
      await browser.close();
      browser = null;
      page = null;
      throw new Error(
        `webgpu_adapter_vendor_mismatch: expected ${requiredVendor}, got ${info.vendor || 'unknown'}`
      );
    }
    return info;
  }

  const deviceInfo = await openSession();

  async function resetSession() {
    if (browser) await browser.close().catch(() => {});
    browser = null;
    page = null;
    const replacement = await openSession();
    if (JSON.stringify(replacement) !== JSON.stringify(deviceInfo)) {
      throw new Error('wgsl_author_device_changed_after_reset');
    }
  }

  return {
    deviceInfo,
    browserArgs,
    executionTimeoutMs,
    async execute(plan, executionOptions = {}) {
      if (plan?.schema !== 'doppler.wgsl-author-execution-plan/v1') {
        throw new Error('WGSL author browser executor requires an execution-plan v1 object.');
      }
      const unavailableFeatures = (plan.requirements?.features || []).filter((feature) => (
        !deviceInfo.availableFeatures.includes(feature)
      ));
      if (unavailableFeatures.length > 0) {
        throw new Error(`wgsl_author_plan_features_unavailable:${unavailableFeatures.join(',')}`);
      }
      for (const limit of plan.requirements?.limits || []) {
        const available = deviceInfo.limits[limit.name];
        if (!Number.isSafeInteger(available) || available < limit.minimum) {
          throw new Error(`wgsl_author_plan_limit_unavailable:${limit.name}:${limit.minimum}`);
        }
      }
      const id = String(executionOptions.id || 'wgsl-author-package');
      const outcome = await settleWithTimeout(
        () => page.evaluate(async (request) => {
          const device = globalThis.__wgslAuthorDevice;
          if (!device) throw new Error('wgsl_author_device_missing');
          const compilation = [];
          const runtimeErrors = [];
          const executedPassIds = [];
          const outputs = {};
          const gpuResources = new Map();
          const staging = [];
          const destroyables = [];
          const moduleObjects = new Map();
          const pipelines = new Map();
          const uncaptured = (event) => {
            runtimeErrors.push(event.error?.message || String(event.error || event));
          };
          const bufferUsageValues = {
            copy_dst: GPUBufferUsage.COPY_DST,
            copy_src: GPUBufferUsage.COPY_SRC,
            storage: GPUBufferUsage.STORAGE,
            uniform: GPUBufferUsage.UNIFORM,
            vertex: GPUBufferUsage.VERTEX,
            index: GPUBufferUsage.INDEX,
          };
          const textureUsageValues = {
            copy_dst: GPUTextureUsage.COPY_DST,
            copy_src: GPUTextureUsage.COPY_SRC,
            texture_binding: GPUTextureUsage.TEXTURE_BINDING,
            storage_binding: GPUTextureUsage.STORAGE_BINDING,
            render_attachment: GPUTextureUsage.RENDER_ATTACHMENT,
          };
          const usageBits = (values, table, label) => values.reduce((bits, name) => {
            if (!Object.hasOwn(table, name)) throw new Error(`${label}_usage_unsupported:${name}`);
            return bits | table[name];
          }, 0);
          const textureDimension = (viewDimension) => {
            if (viewDimension === '1d') return '1d';
            if (viewDimension === '3d') return '3d';
            return '2d';
          };
          const textureView = (resource) => resource.object.createView({
            dimension: resource.descriptor.dimension,
          });
          const bindableResource = (resource) => {
            if (resource.kind.endsWith('_buffer')) return { buffer: resource.object };
            if (resource.kind === 'sampler') return resource.object;
            return textureView(resource);
          };
          const bindGroups = (pass, pipeline) => {
            const groups = new Map();
            for (const binding of pass.bindings) {
              if (!groups.has(binding.group)) groups.set(binding.group, []);
              const resource = gpuResources.get(binding.resourceId);
              if (!resource) throw new Error(`binding_resource_missing:${binding.resourceId}`);
              groups.get(binding.group).push({
                binding: binding.binding,
                resource: bindableResource(resource),
              });
            }
            return [...groups.entries()].sort(([left], [right]) => left - right).map(([group, entries]) => ({
              group,
              bindGroup: device.createBindGroup({
                label: `${request.id}-${pass.id}-group-${group}`,
                layout: pipeline.getBindGroupLayout(group),
                entries: entries.sort((left, right) => left.binding - right.binding),
              }),
            }));
          };
          const setBindGroups = (encoder, groups) => {
            for (const group of groups) encoder.setBindGroup(group.group, group.bindGroup);
          };
          device.addEventListener('uncapturederror', uncaptured);
          device.pushErrorScope('out-of-memory');
          device.pushErrorScope('validation');
          try {
            for (const module of request.plan.modules) {
              const object = device.createShaderModule({
                label: `${request.id}-${module.id}`,
                code: module.wgsl,
              });
              const info = await object.getCompilationInfo();
              const messages = [...info.messages].map((message) => ({
                type: message.type,
                message: message.message,
                lineNum: message.lineNum,
                linePos: message.linePos,
                offset: message.offset,
                length: message.length,
              }));
              compilation.push({ moduleId: module.id, messages });
              moduleObjects.set(module.id, object);
            }
            const compilationFailed = compilation.some((entry) => (
              entry.messages.some((message) => message.type === 'error')
            ));
            if (!compilationFailed) {
              for (const resource of request.plan.resources) {
                if (resource.kind.endsWith('_buffer')) {
                  const object = device.createBuffer({
                    label: `${request.id}-${resource.id}`,
                    size: resource.descriptor.byteLength,
                    usage: usageBits(resource.usage, bufferUsageValues, 'buffer'),
                    mappedAtCreation: true,
                  });
                  const mapped = new Uint8Array(object.getMappedRange());
                  mapped.fill(0);
                  if (resource.initialization.kind === 'bytes') {
                    mapped.set(resource.initialization.bytes);
                  }
                  object.unmap();
                  gpuResources.set(resource.id, { ...resource, object });
                  destroyables.push(object);
                } else if (resource.kind === 'sampler') {
                  const { samplerType: _samplerType, ...descriptor } = resource.descriptor;
                  if (descriptor.compare === null) delete descriptor.compare;
                  const object = device.createSampler(descriptor);
                  gpuResources.set(resource.id, { ...resource, object });
                } else {
                  const descriptor = resource.descriptor;
                  if (descriptor.sampleCount !== 1) {
                    throw new Error(`multisampled_resource_not_supported:${resource.id}`);
                  }
                  const object = device.createTexture({
                    label: `${request.id}-${resource.id}`,
                    size: descriptor.size,
                    mipLevelCount: descriptor.mipLevelCount,
                    sampleCount: descriptor.sampleCount,
                    dimension: textureDimension(descriptor.dimension),
                    format: descriptor.format,
                    usage: usageBits(resource.usage, textureUsageValues, 'texture'),
                  });
                  gpuResources.set(resource.id, { ...resource, object });
                  destroyables.push(object);
                  const bytes = resource.initialization.kind === 'bytes'
                    ? resource.initialization.bytes
                    : new Array(descriptor.copy.tightByteLength).fill(0);
                  device.queue.writeTexture(
                    { texture: object },
                    Uint8Array.from(bytes),
                    {
                      bytesPerRow: descriptor.copy.tightBytesPerRow,
                      rowsPerImage: descriptor.copy.blockRows,
                    },
                    descriptor.size
                  );
                }
              }
              for (const pass of request.plan.passes) {
                const module = moduleObjects.get(pass.moduleId);
                if (pass.kind === 'compute') {
                  pipelines.set(pass.id, await device.createComputePipelineAsync({
                    label: `${request.id}-${pass.id}`,
                    layout: 'auto',
                    compute: {
                      module,
                      entryPoint: pass.entryPoints.compute,
                      constants: pass.constants,
                    },
                  }));
                } else {
                  const vertexLayouts = [];
                  for (const vertexBuffer of pass.vertexBuffers) {
                    const resource = gpuResources.get(vertexBuffer.resourceId);
                    vertexLayouts[vertexBuffer.slot] = {
                      arrayStride: resource.descriptor.arrayStride,
                      stepMode: resource.descriptor.stepMode,
                      attributes: resource.descriptor.attributes,
                    };
                  }
                  for (let index = 0; index < vertexLayouts.length; index += 1) {
                    if (vertexLayouts[index] === undefined) vertexLayouts[index] = null;
                  }
                  const primitive = { ...pass.primitive };
                  if (primitive.stripIndexFormat === null) delete primitive.stripIndexFormat;
                  pipelines.set(pass.id, await device.createRenderPipelineAsync({
                    label: `${request.id}-${pass.id}`,
                    layout: 'auto',
                    vertex: {
                      module,
                      entryPoint: pass.entryPoints.vertex,
                      constants: pass.constants,
                      buffers: vertexLayouts,
                    },
                    fragment: {
                      module,
                      entryPoint: pass.entryPoints.fragment,
                      constants: pass.constants,
                      targets: pass.targets.map((target) => ({
                        format: target.format,
                        blend: target.blend === null ? undefined : target.blend,
                        writeMask: target.writeMask,
                      })),
                    },
                    primitive,
                    multisample: pass.multisample,
                  }));
                }
              }
              const commandEncoder = device.createCommandEncoder({
                label: `${request.id}-encoder`,
              });
              for (const pass of request.plan.passes) {
                const pipeline = pipelines.get(pass.id);
                const groups = bindGroups(pass, pipeline);
                if (pass.kind === 'compute') {
                  const encoder = commandEncoder.beginComputePass({
                    label: `${request.id}-${pass.id}`,
                  });
                  encoder.setPipeline(pipeline);
                  setBindGroups(encoder, groups);
                  encoder.setViewport(
                    pass.viewport.x,
                    pass.viewport.y,
                    pass.viewport.width,
                    pass.viewport.height,
                    pass.viewport.minDepth,
                    pass.viewport.maxDepth
                  );
                  encoder.setScissorRect(
                    pass.scissor.x,
                    pass.scissor.y,
                    pass.scissor.width,
                    pass.scissor.height
                  );
                  encoder.dispatchWorkgroups(...pass.dispatch);
                  encoder.end();
                } else {
                  const colorAttachments = pass.targets.map((target) => {
                    const resource = gpuResources.get(target.resourceId);
                    const attachment = {
                      view: textureView(resource),
                      loadOp: target.loadOp,
                      storeOp: target.storeOp,
                    };
                    if (target.clearValue !== null) {
                      const [r, g, b, a] = target.clearValue;
                      attachment.clearValue = { r, g, b, a };
                    }
                    return attachment;
                  });
                  const encoder = commandEncoder.beginRenderPass({
                    label: `${request.id}-${pass.id}`,
                    colorAttachments,
                  });
                  encoder.setPipeline(pipeline);
                  setBindGroups(encoder, groups);
                  for (const vertexBuffer of pass.vertexBuffers) {
                    encoder.setVertexBuffer(
                      vertexBuffer.slot,
                      gpuResources.get(vertexBuffer.resourceId).object
                    );
                  }
                  if (pass.draw.kind === 'indexed') {
                    const indexResource = gpuResources.get(pass.indexBuffer.resourceId);
                    encoder.setIndexBuffer(
                      indexResource.object,
                      indexResource.descriptor.indexFormat
                    );
                    encoder.drawIndexed(
                      pass.draw.indexCount,
                      pass.draw.instanceCount,
                      pass.draw.firstIndex,
                      pass.draw.baseVertex,
                      pass.draw.firstInstance
                    );
                  } else {
                    encoder.draw(
                      pass.draw.vertexCount,
                      pass.draw.instanceCount,
                      pass.draw.firstVertex,
                      pass.draw.firstInstance
                    );
                  }
                  encoder.end();
                }
                executedPassIds.push(pass.id);
              }
              for (const outputId of request.plan.outputs) {
                const resource = gpuResources.get(outputId);
                if (resource.kind.endsWith('_buffer')) {
                  const object = device.createBuffer({
                    label: `${request.id}-${outputId}-readback`,
                    size: resource.descriptor.byteLength,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                  });
                  commandEncoder.copyBufferToBuffer(
                    resource.object,
                    0,
                    object,
                    0,
                    resource.descriptor.byteLength
                  );
                  staging.push({
                    id: outputId,
                    kind: 'buffer',
                    object,
                    byteLength: resource.descriptor.byteLength,
                  });
                  destroyables.push(object);
                } else {
                  const copy = resource.descriptor.copy;
                  const paddedBytesPerRow = Math.ceil(copy.tightBytesPerRow / 256) * 256;
                  const byteLength = paddedBytesPerRow
                    * copy.blockRows
                    * resource.descriptor.size.depthOrArrayLayers;
                  const object = device.createBuffer({
                    label: `${request.id}-${outputId}-readback`,
                    size: byteLength,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                  });
                  commandEncoder.copyTextureToBuffer(
                    { texture: resource.object },
                    {
                      buffer: object,
                      bytesPerRow: paddedBytesPerRow,
                      rowsPerImage: copy.blockRows,
                    },
                    resource.descriptor.size
                  );
                  staging.push({
                    id: outputId,
                    kind: 'texture',
                    object,
                    byteLength,
                    paddedBytesPerRow,
                    tightBytesPerRow: copy.tightBytesPerRow,
                    blockRows: copy.blockRows,
                    depthOrArrayLayers: resource.descriptor.size.depthOrArrayLayers,
                  });
                  destroyables.push(object);
                }
              }
              device.queue.submit([commandEncoder.finish()]);
              await device.queue.onSubmittedWorkDone();
              for (const entry of staging) {
                await entry.object.mapAsync(GPUMapMode.READ);
                const mapped = new Uint8Array(entry.object.getMappedRange()).slice();
                if (entry.kind === 'buffer') {
                  outputs[entry.id] = { kind: entry.kind, bytes: [...mapped] };
                } else {
                  const bytes = [];
                  for (let layer = 0; layer < entry.depthOrArrayLayers; layer += 1) {
                    for (let row = 0; row < entry.blockRows; row += 1) {
                      const start = (layer * entry.blockRows + row) * entry.paddedBytesPerRow;
                      bytes.push(...mapped.slice(start, start + entry.tightBytesPerRow));
                    }
                  }
                  outputs[entry.id] = { kind: entry.kind, bytes };
                }
                entry.object.unmap();
              }
            }
          } catch (error) {
            runtimeErrors.push(error?.message || String(error));
          } finally {
            const validationError = await device.popErrorScope();
            const outOfMemoryError = await device.popErrorScope();
            if (validationError) runtimeErrors.push(validationError.message);
            if (outOfMemoryError) runtimeErrors.push(outOfMemoryError.message);
            device.removeEventListener('uncapturederror', uncaptured);
            for (const object of destroyables.reverse()) object.destroy();
          }
          return { compilation, runtimeErrors, executedPassIds, outputs };
        }, { id, plan }),
        executionTimeoutMs
      );
      if (outcome.status === 'fulfilled') {
        return summarizeExecution(id, plan, deviceInfo, browserArgs, outcome.value);
      }
      const reason = outcome.status === 'timed_out'
        ? `WGSL author execution timed out after ${executionTimeoutMs}ms.`
        : `WGSL author browser exception: ${outcome.error?.message || String(outcome.error)}`;
      await resetSession();
      return summarizeExecution(id, plan, deviceInfo, browserArgs, {
        compilation: [],
        runtimeErrors: [reason],
        executedPassIds: [],
        outputs: {},
      });
    },
    async close() {
      if (browser) await browser.close();
      browser = null;
      page = null;
    },
  };
}

export { summarizeExecution as summarizeWgslAuthorBrowserExecution };
