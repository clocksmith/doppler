import { computeCanonicalSha256, hashBytesSha256 } from '../../formats/canonical-hash.js';
import { createDeviceAvailabilityCheck } from './resource-binder.js';
function resolveGpuDevice(devicePort) {
  const device = typeof devicePort?.getDevice === 'function' ? devicePort.getDevice() : devicePort?.gpuDevice ?? devicePort;
  if (!device || typeof device.createCommandEncoder !== 'function') {
    throw new Error('CommandExecutor requires a physical GPUDevice.');
  }
  return device;
}

function normalizeWorkgroups(value) {
  if (!Array.isArray(value) || value.length < 1 || value.length > 3) {
    throw new Error('Dispatch command workgroups must be a one-to-three element integer array.');
  }
  const result = [value[0], value[1] ?? 1, value[2] ?? 1];
  if (result.some((entry) => !Number.isInteger(entry) || entry < 1)) {
    throw new Error('Dispatch workgroups must be positive integers.');
  }
  return result;
}

function assertNotAborted(signal, submission = 'not-submitted') {
  if (signal?.aborted) {
    throw Object.assign(new Error('Command execution aborted.', { cause: signal.reason }), {
      name: 'AbortError', code: 'COMMAND_ABORTED', submission,
    });
  }
}

export function createCommandExecutor(devicePort, resourceBinder, program = null) {
  const device = resolveGpuDevice(devicePort);
  const pipelineTasks = new Map();
  const moduleIdentities = new WeakMap();
  const assertDeviceAvailable = createDeviceAvailabilityCheck(device);

  function assertExecutable(signal, submission = 'not-submitted') {
    assertNotAborted(signal, submission);
    try {
      assertDeviceAvailable();
      if (resolveGpuDevice(devicePort) !== device) throw new Error('GPU device replaced.');
    } catch (cause) {
      pipelineTasks.clear();
      throw Object.assign(new Error('Command execution GPU device lost or replaced.', { cause }), {
        code: 'COMMAND_DEVICE_LOST', submission,
      });
    }
  }

  async function resolvePipeline(command, module) {
    let identity = moduleIdentities.get(module);
    if (identity?.source !== module.source) {
      identity = { source: module.source, digest: hashBytesSha256(new TextEncoder().encode(module.source)) };
      moduleIdentities.set(module, identity);
    }
    const key = computeCanonicalSha256({
      moduleId: module.id,
      sourceHash: module.sourceHash,
      sourceDigest: identity.digest,
      entry: command.entry ?? module.entry,
      constants: command.constants ?? {},
    });
    if (!pipelineTasks.has(key)) {
      const shaderModule = device.createShaderModule({
        label: `doppler-capsule:${module.id}`,
        code: module.source,
      });
      const descriptor = {
        label: `doppler-capsule:${command.id ?? module.id}`,
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: command.entry ?? module.entry,
          constants: command.constants ?? {},
        },
      };
      const task = Promise.resolve().then(() => {
        assertExecutable(null);
        return typeof device.createComputePipelineAsync === 'function'
          ? device.createComputePipelineAsync(descriptor)
          : device.createComputePipeline(descriptor);
      }).catch(error => {
        if (pipelineTasks.get(key) === task) pipelineTasks.delete(key);
        throw error;
      });
      pipelineTasks.set(key, task);
    }
    return pipelineTasks.get(key);
  }

  async function executeDispatch(command, modules, signal) {
    assertExecutable(signal);
    const module = modules.get(command.moduleId);
    if (!module?.source) throw new Error(`Dispatch command references unavailable WGSL module "${command.moduleId}".`);
    const pipeline = await resolvePipeline(command, module);
    assertExecutable(signal);
    const entries = (command.bindings || []).map((binding) => {
      const slot = resourceBinder.getSlot(binding.slotId);
      const buffer = slot?.buffer ?? slot?.resource?.buffer ?? slot?.resource;
      if (!buffer) throw new Error(`Dispatch binding references unbound GPU slot "${binding.slotId}".`);
      return {
        binding: binding.binding,
        resource: {
          buffer,
          offset: binding.offset ?? 0,
          ...(binding.size == null ? {} : { size: binding.size }),
        },
      };
    });
    assertExecutable(signal);
    const bindGroup = device.createBindGroup({
      label: `doppler-capsule:${command.id ?? command.moduleId}:bindings`,
      layout: pipeline.getBindGroupLayout(command.group ?? 0),
      entries,
    });
    const encoder = device.createCommandEncoder({ label: `doppler-capsule:${command.id ?? command.moduleId}` });
    const pass = encoder.beginComputePass({ label: `doppler-capsule:${command.id ?? command.moduleId}:compute` });
    pass.setPipeline(pipeline);
    pass.setBindGroup(command.group ?? 0, bindGroup);
    const [x, y, z] = normalizeWorkgroups(command.workgroups);
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
    const commands = encoder.finish();
    assertExecutable(signal);
    device.queue.submit([commands]);
    if (command.waitForCompletion === true) await device.queue.onSubmittedWorkDone();
    assertExecutable(signal, 'submitted');
    return {
      kind: 'dispatch', moduleId: module.id, workgroups: [x, y, z],
      outcome: command.waitForCompletion === true ? 'completed' : 'submitted',
    };
  }

  async function executeProgramPhase(phase, command, options) {
    if (!program || typeof program.executePhase !== 'function') {
      throw new Error(`Program phase "${phase}" requires an injected sealed program executor.`);
    }
    if (command.phase !== phase) throw new Error(`Program phase command "${command.phase}" cannot execute in "${phase}".`);
    if (program.executionGraphHash !== command.executionGraphHash) {
      throw new Error(`Program phase "${phase}" execution graph digest mismatch.`);
    }
    return program.executePhase(phase, {
      declaredStepIds: command.declaredStepIds,
      context: options.context,
      signal: options.signal,
    });
  }

  return {
    async executePhase(phase, commands = [], options = {}) {
      if (!Array.isArray(commands) || commands.length === 0) {
        throw new Error(`TargetPlan phase "${phase}" has no declared commands.`);
      }
      const results = [];
      for (const command of commands) {
        assertNotAborted(options.signal);
        if (command.kind === 'dispatch') {
          results.push(await executeDispatch(command, options.modules ?? new Map(), options.signal));
        } else if (command.kind === 'program-phase') {
          results.push(await executeProgramPhase(phase, command, options));
        } else {
          throw new Error(`TargetPlan phase "${phase}" contains unsupported command kind "${command?.kind}".`);
        }
      }
      return { ok: true, phase, commandCount: commands.length, results };
    },

    clearPipelineCache() {
      pipelineTasks.clear();
    },
  };
}
