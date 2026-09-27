import { getKernelRegistry } from './kernel-configs.js';
import { getCachedPipeline, getPipelineFast, getPipelineBindGroupLayout } from './pipeline-cache.js';
import { getDevice } from '../device.js';
import { dispatchKernel, dispatchIndirect, recordDispatchIndirect } from './dispatch.js';
import { createKernelUniformBuffer } from './uniform-utils.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';

const dataBindingsByConfig = new WeakMap();

function getDataBindings(config) {
  let cached = dataBindingsByConfig.get(config);
  if (cached) {
    return cached;
  }
  cached = config.bindings
    .filter(b => b.type !== 'uniform')
    .slice()
    .sort((a, b) => a.index - b.index);
  dataBindingsByConfig.set(config, cached);
  return cached;
}

export async function unifiedKernelWrapper(
  opName,
  target,
  variant,
  bindings,
  uniforms,
  workgroups,
  constants = null,
  extraBindings = null,
  dispatchLabel = null,
  signal = null
) {
  signal?.throwIfAborted();
  const device = target?.device ?? (target?.createCommandEncoder ? target : getDevice());
  const recorder = target && typeof target.beginComputePass === 'function' ? target : null;
  const registry = getKernelRegistry();
  const config = registry.getKernelConfig(opName, variant);
  const validate = registry.getKernelValidator(opName, variant);
  const pipeline = getCachedPipeline(opName, variant, constants, device)
    ?? await getPipelineFast(opName, variant, null, constants, device);

  signal?.throwIfAborted();
  const bindGroupEntries = [];

  const dataBindings = getDataBindings(config);

  if (bindings.length !== dataBindings.length) {
    throw new Error(
      `Kernel "${opName}/${variant}" expected ${dataBindings.length} bindings ` +
      `(excluding uniforms) but got ${bindings.length}`
    );
  }

  validate?.({ operation: opName, variant, bindings, uniforms, workgroups, constants, extraBindings });

  for (let i = 0; i < bindings.length; i++) {
    const binding = bindings[i];
    const bindingConfig = dataBindings[i];
    let index = bindingConfig.index;

    // Some variants change output binding index (e.g. gather f16 output uses binding 4).
    if (bindingConfig.name === 'output' && config.variantMetadata?.outputBinding != null) {
      index = config.variantMetadata.outputBinding;
    }

    const buffer = binding?.buffer || binding;
    const isGpuBuffer = buffer && (
      typeof GPUBuffer === 'undefined'
        ? true
        : buffer instanceof GPUBuffer
    );
    if (!isGpuBuffer) {
      const bindingLabel = binding?.label ?? buffer?.label ?? 'unknown';
      const bufferType = buffer === null ? 'null' : buffer === undefined ? 'undefined' : buffer.constructor?.name || typeof buffer;
      throw new Error(
        `Kernel "${opName}/${variant}" binding "${bindingConfig.name}" (index ${index}) requires a GPUBuffer ` +
        `(label=${bindingLabel}, type=${bufferType}).`
      );
    }

    bindGroupEntries.push({
      binding: index,
      resource: { buffer }
    });
  }

  // Append extra bindings not tracked in registry (e.g. OUTPUT_PRENORM residual_sum_output)
  if (extraBindings) {
    for (const extra of extraBindings) {
      const buf = extra.buffer?.buffer || extra.buffer;
      bindGroupEntries.push({
        binding: extra.binding,
        resource: { buffer: buf },
      });
    }
  }

  let uniformBuffer = null;
  try {
    if (config.uniforms !== null) {
      uniformBuffer = createKernelUniformBuffer(`${opName}_uniforms`, config, uniforms, recorder, device);
      const uniformBinding = config.bindings.find(binding => binding.type === 'uniform');
      if (!uniformBinding) throw new Error(`Kernel ${opName}/${variant} declares uniforms without a binding.`);
      bindGroupEntries.unshift({ binding: uniformBinding.index, resource: { buffer: uniformBuffer } });
    }
    const bindGroup = device.createBindGroup({
      label: `${opName}_bind_group`,
      layout: getPipelineBindGroupLayout(pipeline, 0),
      entries: bindGroupEntries,
    });

    signal?.throwIfAborted();
    const label = typeof dispatchLabel === 'string' && dispatchLabel.length > 0
      ? dispatchLabel
      : opName;
    if (workgroups && typeof workgroups === 'object' && workgroups.indirectBuffer) {
      const indirectOffset = workgroups.indirectOffset ?? 0;
      if (recorder) {
        recordDispatchIndirect(recorder, pipeline, bindGroup, workgroups.indirectBuffer, indirectOffset, label);
      } else {
        dispatchIndirect(device, pipeline, bindGroup, workgroups.indirectBuffer, indirectOffset, label);
      }
    } else {
      dispatchKernel(recorder ?? device, pipeline, bindGroup, workgroups, label);
    }
  } catch (error) {
    if (!recorder) {
      uniformBuffer?.destroy();
    }
    throw error;
  }

  if (!recorder && uniformBuffer) {
    device.queue.onSubmittedWorkDone()
      .then(() => {
        uniformBuffer.destroy();
      })
      .catch(() => {
        uniformBuffer.destroy();
      });
  }

  return true;
}

// The wrapper retains shape semantics; this executor owns only allocated output
// rollback. Recorded commands retain failed outputs until recorder cleanup.
export async function withKernelOutput(target, supplied, bytes, label, execute) {
  const output = supplied ?? acquireBuffer(bytes, undefined, label);
  try { return await execute(output); }
  catch (error) {
    if (!supplied) {
      if (target && typeof target.beginComputePass === 'function') target.trackTemporaryBuffer(output);
      else releaseBuffer(output);
    }
    throw error;
  }
}
