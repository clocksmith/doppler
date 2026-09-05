import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';
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

export function createCommandExecutor(devicePort, resourceBinder, program = null) {
  const device = resolveGpuDevice(devicePort);
  const pipelineTasks = new Map();

  async function resolvePipeline(command, module) {
    const key = computeCanonicalSha256({
      moduleId: module.id,
      sourceHash: module.sourceHash,
      entry: command.entry ?? module.entry,
      constants: command.constants ?? {},
    });
    if (!pipelineTasks.has(key)) {
      const shaderModule = device.createShaderModule({
        label: `doppler-pack:${module.id}`,
        code: module.source,
      });
      const descriptor = {
        label: `doppler-pack:${command.id ?? module.id}`,
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: command.entry ?? module.entry,
          constants: command.constants ?? {},
        },
      };
      pipelineTasks.set(
        key,
        typeof device.createComputePipelineAsync === 'function'
          ? device.createComputePipelineAsync(descriptor)
          : Promise.resolve(device.createComputePipeline(descriptor))
      );
    }
    return pipelineTasks.get(key);
  }

  async function executeDispatch(command, modules) {
    const module = modules.get(command.moduleId);
    if (!module?.source) throw new Error(`Dispatch command references unavailable WGSL module "${command.moduleId}".`);
    const pipeline = await resolvePipeline(command, module);
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
    const bindGroup = device.createBindGroup({
      label: `doppler-pack:${command.id ?? command.moduleId}:bindings`,
      layout: pipeline.getBindGroupLayout(command.group ?? 0),
      entries,
    });
    const encoder = device.createCommandEncoder({ label: `doppler-pack:${command.id ?? command.moduleId}` });
    const pass = encoder.beginComputePass({ label: `doppler-pack:${command.id ?? command.moduleId}:compute` });
    pass.setPipeline(pipeline);
    pass.setBindGroup(command.group ?? 0, bindGroup);
    const [x, y, z] = normalizeWorkgroups(command.workgroups);
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
    device.queue.submit([encoder.finish()]);
    if (command.waitForCompletion === true) await device.queue.onSubmittedWorkDone();
    return { kind: 'dispatch', moduleId: module.id, workgroups: [x, y, z] };
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
        if (options.signal?.aborted) throw new Error(`Command execution aborted during phase "${phase}".`);
        if (command.kind === 'dispatch') {
          results.push(await executeDispatch(command, options.modules ?? new Map()));
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
