import { createResourceBinder } from '../../../client/runtime/resource-binder.js';
import { createCommandExecutor } from '../../../client/runtime/command-executor.js';
import { validateForecastManifest } from '../../../config/forecast-manifest.js';
import { readbackBuffer } from '../../../gpu/readback-buffer.js';
import { log } from '../../../debug/index.js';

export function createForecastProgramFactory(devicePort) {
  const device = typeof devicePort?.getDevice === 'function' ? devicePort.getDevice() : devicePort?.gpuDevice ?? devicePort;
  if (typeof device?.createBuffer !== 'function') throw new Error('Forecast program requires an explicit GPUDevice.');
  return async function createForecastProgram({ pack, targetPlan, artifactStore }) {
    const manifestArtifact = pack.artifacts.find(a => a.artifactId === pack.program.manifestArtifactId);
    const manifest = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(await artifactStore.readArtifact(manifestArtifact)));
    const contract = validateForecastManifest(manifest, pack, targetPlan);
    const binder = createResourceBinder(device);
    const executor = createCommandExecutor(device, binder);
    let closed = false;
    let active = null;
    let loss = null;
    device.lost.then(info => { loss = info; });
    try {
      binder.bindSlots(targetPlan.memoryLayout, {});
      for (const artifactId of new Set(manifest.uploads.map(upload => upload.artifactId))) {
        const artifact = pack.artifacts.find(a => a.artifactId === artifactId);
        const bytes = await artifactStore.readArtifact(artifact);
        for (const upload of manifest.uploads.filter(entry => entry.artifactId === artifactId)) {
          binder.writeSlot(upload.slotId, bytes.subarray(upload.offsetBytes, upload.offsetBytes + upload.sizeBytes));
        }
      }
      const modules = new Map();
      for (const module of pack.wgslModules) {
        const artifact = pack.artifacts.find(a => a.artifactId === module.sourceArtifactId);
        const source = new TextDecoder('utf-8', { fatal: true }).decode(await artifactStore.readArtifact(artifact));
        modules.set(module.id, { ...module, source });
      }
      log.info('Forecast', `Declared ${targetPlan.targetId} lane: activation=${targetPlan.dtypes.activation}, weight=${targetPlan.dtypes.weight}.`);
      async function execute(request, signal) {
        if (!Array.isArray(request.context) || request.context.length < 1 || request.context.length > contract.contextLength
          || request.context.some(value => !Number.isFinite(value) || !Number.isFinite(Math.fround(value)))
          || !Number.isSafeInteger(request.horizon) || request.horizon < 1 || request.horizon > contract.predictionLength) {
          throw new Error('Forecast request exceeds its qualified numeric context/horizon envelope.');
        }
        if (signal?.aborted) throw signal.reason ?? new Error('Forecast cancelled.');
        // Host preprocessing: F32 encoding, right alignment, explicit absence mask.
        const input = new Float32Array(contract.contextLength);
        const mask = new Float32Array(contract.contextLength);
        const start = contract.contextLength - request.context.length;
        input.set(request.context, start);
        mask.fill(1, start);
        binder.writeSlot(contract.inputSlot, input);
        binder.writeSlot(contract.maskSlot, mask);
        binder.writeSlot(contract.requestSlot, new Uint32Array([request.horizon, 0, 0, 0]));
        device.pushErrorScope('validation');
        try {
          await executor.executePhase('forecast', targetPlan.phases.forecast, { modules, signal });
          const bytes = await readbackBuffer(device, binder.getSlot(contract.outputSlot).buffer,
            request.horizon * contract.quantiles.length * 4);
          if (signal?.aborted) throw signal.reason ?? new Error('Forecast cancelled.');
          if (loss) throw new Error(`Forecast GPU device lost: ${loss.message}`);
          const values = Array.from(new Float32Array(bytes));
          if (values.some(value => !Number.isFinite(value))) throw new Error('Forecast produced a non-finite value.');
          return { horizon: request.horizon, quantileLevels: [...contract.quantiles], layout: contract.outputLayout, values };
        } finally {
          const validationError = await device.popErrorScope();
          if (validationError) throw new Error(`Forecast GPU validation failed: ${validationError.message}`);
        }
      }
      return {
        executionGraphHash: targetPlan.executionGraphHash,
        async forecast(request, options) {
          if (closed || loss) throw new Error('Forecast program is closed or its device is lost.');
          if (active) throw new Error('Forecast session is already executing a job.');
          active = execute(request, options?.signal);
          try { return await active; } finally { active = null; }
        },
        async close() {
          if (closed) return;
          closed = true;
          try { await active; } finally { binder.releaseAll(); executor.clearPipelineCache(); }
        },
      };
    } catch (error) {
      binder.releaseAll(); executor.clearPipelineCache(); throw error;
    }
  };
}
