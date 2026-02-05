import { getDevice, setDevice, getKernelCapabilities } from '../../gpu/device.js';
import { createTensor, tensorBytes } from '../../gpu/tensor.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../memory/buffer-pool.js';
import {
  runEnergyEval,
  runEnergyUpdate,
  runEnergyQuintelUpdate,
  runEnergyQuintelReduce,
} from '../../gpu/kernels/index.js';
import { WORKGROUP_SIZES } from '../../gpu/kernels/constants.js';
import { computeArrayStats } from '../../debug/stats.js';
import { log, trace, applyDebugConfig, setGPUDevice } from '../../debug/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../config/runtime.js';
import { DEFAULT_ENERGY_CONFIG } from '../../config/schema/energy.schema.js';
import { f32ToF16Array, f16ToF32Array } from '../kv-cache/types.js';
import { registerPipeline } from '../pipeline/registry.js';
import { mergeQuintelConfig, runQuintelEnergyLoop } from './quintel.js';
import { runVliwEnergyLoop } from './vliw.js';

function createRng(seed) {
  let state = seed >>> 0;
  if (!state) state = 0x6d2b79f5;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateRandomArray(count, mode, seed, scale) {
  const out = new Float32Array(count);
  if (mode === 'zeros' || mode === 'baseline') {
    return out;
  }
  const rand = createRng(seed);
  const safeScale = Number.isFinite(scale) ? scale : 1.0;
  for (let i = 0; i < count; i++) {
    if (mode === 'uniform') {
      out[i] = (rand() * 2 - 1) * safeScale;
    } else {
      const u = rand();
      const v = rand();
      const z = Math.sqrt(-2.0 * Math.log(Math.max(u, 1e-6))) * Math.cos(2.0 * Math.PI * v);
      out[i] = z * safeScale;
    }
  }
  return out;
}

function resolveShape(config) {
  if (Array.isArray(config.shape) && config.shape.length > 0) {
    return config.shape.map((value) => Math.max(1, Math.floor(value)));
  }
  const width = Math.max(1, Math.floor(config.state.width));
  const height = Math.max(1, Math.floor(config.state.height));
  const channels = Math.max(1, Math.floor(config.state.channels));
  return [height, width, channels];
}

function mergeEnergyConfig(base, override) {
  if (!override) {
    return {
      ...base,
      quintel: mergeQuintelConfig(base.quintel, null),
    };
  }
  return {
    ...base,
    ...override,
    problem: override.problem ?? base.problem,
    state: { ...base.state, ...override.state },
    init: { ...base.init, ...override.init },
    target: { ...base.target, ...override.target },
    loop: { ...base.loop, ...override.loop },
    diagnostics: { ...base.diagnostics, ...override.diagnostics },
    quintel: mergeQuintelConfig(base.quintel, override.quintel),
  };
}

function diffEnergySection(section, defaults) {
  if (!section) return {};
  const out = {};
  for (const [key, value] of Object.entries(section)) {
    if (value === undefined) continue;
    if (!Object.is(value, defaults?.[key])) {
      out[key] = value;
    }
  }
  return out;
}

function diffQuintelConfig(config, defaults) {
  if (!config) return {};
  const out = {};
  if (typeof config.backend === 'string' && config.backend !== defaults?.backend) {
    out.backend = config.backend;
  }
  if (Number.isFinite(config.size) && !Object.is(config.size, defaults?.size)) {
    out.size = config.size;
  }
  if (Number.isFinite(config.countTarget) && !Object.is(config.countTarget, defaults?.countTarget)) {
    out.countTarget = config.countTarget;
  }
  if (Number.isFinite(config.centerTarget) && !Object.is(config.centerTarget, defaults?.centerTarget)) {
    out.centerTarget = config.centerTarget;
  }
  const rules = diffEnergySection(config.rules, defaults?.rules);
  const weights = diffEnergySection(config.weights, defaults?.weights);
  const clamp = diffEnergySection(config.clamp, defaults?.clamp);
  if (Object.keys(rules).length) out.rules = rules;
  if (Object.keys(weights).length) out.weights = weights;
  if (Object.keys(clamp).length) out.clamp = clamp;
  return out;
}

function diffEnergyConfig(config, defaults) {
  if (!config) return {};
  const state = diffEnergySection(config.state, defaults.state);
  const init = diffEnergySection(config.init, defaults.init);
  const target = diffEnergySection(config.target, defaults.target);
  const loop = diffEnergySection(config.loop, defaults.loop);
  const diagnostics = diffEnergySection(config.diagnostics, defaults.diagnostics);
  const quintel = diffQuintelConfig(config.quintel, defaults.quintel);
  return {
    ...(config.problem && config.problem !== defaults.problem ? { problem: config.problem } : {}),
    ...(Object.keys(state).length ? { state } : {}),
    ...(Object.keys(init).length ? { init } : {}),
    ...(Object.keys(target).length ? { target } : {}),
    ...(Object.keys(loop).length ? { loop } : {}),
    ...(Object.keys(diagnostics).length ? { diagnostics } : {}),
    ...(Object.keys(quintel).length ? { quintel } : {}),
  };
}

function resolveEnergyRuntime(manifest, runtimeConfig) {
  const modelEnergy = manifest?.energy ?? manifest?.config?.energy ?? {};
  const runtimeEnergy = runtimeConfig?.inference?.energy ?? {};
  const runtimeOverrides = diffEnergyConfig(runtimeEnergy, DEFAULT_ENERGY_CONFIG);
  const merged = mergeEnergyConfig(
    mergeEnergyConfig(DEFAULT_ENERGY_CONFIG, modelEnergy),
    runtimeOverrides
  );
  merged.shape = modelEnergy.shape ?? null;
  return merged;
}

async function createEnergyTensor(device, data, dtype, shape, label) {
  const byteLength = data.byteLength;
  const alignedSize = Math.ceil(byteLength / 4) * 4;
  const buffer = acquireBuffer(alignedSize, undefined, label);

  let payload = data;
  if (alignedSize !== byteLength) {
    const padded = new Uint8Array(alignedSize);
    const view = data instanceof ArrayBuffer
      ? new Uint8Array(data)
      : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    padded.set(view);
    payload = padded;
  }

  device.queue.writeBuffer(buffer, 0, payload);
  const tensor = createTensor(buffer, dtype, shape, label);
  const expectedBytes = tensorBytes(shape, dtype);
  if (expectedBytes !== byteLength) {
    log.warn('Energy', `${label} byte length mismatch: expected ${expectedBytes}, got ${byteLength}`);
  }
  return tensor;
}

async function readTensorToFloat32(tensor) {
  const byteLength = tensorBytes(tensor.shape, tensor.dtype);
  const data = await readBuffer(tensor.buffer, byteLength);
  if (tensor.dtype === 'f16') {
    return f16ToF32Array(new Uint16Array(data));
  }
  return new Float32Array(data);
}

export class EnergyPipeline {
  runtimeConfig = null;
  manifest = null;
  stats = {};
  baseUrl = null;
  _onProgress = null;

  async initialize(contexts = {}) {
    if (contexts.runtimeConfig) {
      this.runtimeConfig = setRuntimeConfig(contexts.runtimeConfig);
    } else {
      this.runtimeConfig = getRuntimeConfig();
    }
    const sharedDebug = this.runtimeConfig.shared?.debug;
    if (sharedDebug) {
      applyDebugConfig(sharedDebug);
    }

    if (contexts.gpu?.device) {
      setDevice(contexts.gpu.device);
      setGPUDevice(contexts.gpu.device);
    } else {
      const device = getDevice();
      if (device) setGPUDevice(device);
    }

    if (contexts.baseUrl) this.baseUrl = contexts.baseUrl;
    if (contexts.onProgress) this._onProgress = contexts.onProgress;
  }

  async loadModel(manifest) {
    if (!manifest || manifest.modelType !== 'energy') {
      throw new Error('Energy pipeline requires an energy model manifest.');
    }
    this.manifest = manifest;
    log.info('Energy', `Loaded energy model "${manifest.modelId}"`);
  }

  getStats() {
    return this.stats;
  }

  getMemoryStats() {
    return {
      used: 0,
      kvCache: null,
    };
  }

  async unload() {
    this.manifest = null;
  }

  async generate(request = {}) {
    const runtimeConfig = resolveEnergyRuntime(this.manifest, this.runtimeConfig);
    const problem = request.problem ?? runtimeConfig.problem ?? 'l2';

    const loopConfig = runtimeConfig.loop;
    const maxSteps = Math.max(1, Math.floor(request.steps ?? loopConfig.maxSteps));
    const minSteps = Math.max(1, Math.floor(loopConfig.minSteps));
    const stepSize = request.stepSize ?? loopConfig.stepSize;
    const gradientScale = request.gradientScale ?? loopConfig.gradientScale;
    const convergenceThreshold = Number.isFinite(request.convergenceThreshold)
      ? request.convergenceThreshold
      : loopConfig.convergenceThreshold;

    const diagnostics = runtimeConfig.diagnostics;
    const readbackEvery = Math.max(1, Math.floor(request.readbackEvery ?? diagnostics.readbackEvery));
    const historyLimit = Math.max(1, Math.floor(diagnostics.historyLimit));
    const traceEvery = Math.max(1, Math.floor(diagnostics.traceEvery));

    if (problem === 'quintel') {
      const quintelConfig = mergeQuintelConfig(runtimeConfig.quintel, request.quintel);
      const size = Math.max(1, Math.floor(quintelConfig.size ?? 5));
      const elementCount = size * size;
      const shape = [size, size, 1];

      const initMode = request.initMode ?? runtimeConfig.init.mode;
      const initSeed = request.seed ?? runtimeConfig.init.seed;
      const initScale = request.initScale ?? runtimeConfig.init.scale;
      const initData = generateRandomArray(elementCount, initMode, initSeed, initScale);

      if (this._onProgress) {
        this._onProgress({ stage: 'energy', percent: 0, message: 'Initializing Quintel state' });
      }

      const loop = {
        maxSteps,
        minSteps,
        stepSize,
        gradientScale,
        convergenceThreshold,
      };
      const diagnosticsConfig = {
        readbackEvery,
        historyLimit,
        traceEvery,
      };

      const runCpu = () => {
        log.info('Energy', 'Quintel backend: CPU');
        const result = runQuintelEnergyLoop({
          state: initData,
          size,
          config: quintelConfig,
          loop,
          diagnostics: diagnosticsConfig,
          onProgress: this._onProgress,
          onTrace: (step, energy, components) => {
            if (!Number.isFinite(energy)) return;
            trace.energy(`step=${step} energy=${energy.toFixed(6)}`, components);
          },
          traceEvery,
        });

        const stateStats = computeArrayStats(result.state);

        this.stats = {
          backend: 'CPU',
          totalTimeMs: result.totalTimeMs,
          steps: result.steps,
          stepTimesMs: result.stepTimesMs,
          energyHistory: result.energyHistory,
          readbackCount: result.energyHistory.length,
          energy: result.energy,
          energyComponents: result.energyComponents,
          stateStats,
        };

        return {
          backend: 'CPU',
          shape,
          dtype: 'f32',
          steps: result.steps,
          energy: result.energy,
          state: result.state,
          energyHistory: result.energyHistory,
          energyComponents: result.energyComponents,
          stateStats,
          totalTimeMs: result.totalTimeMs,
          problem: 'quintel',
        };
      };

      const device = getDevice();
      const requestedBackend = typeof quintelConfig.backend === 'string'
        ? quintelConfig.backend.toLowerCase()
        : 'cpu';
      const resolvedBackend = requestedBackend === 'auto'
        ? (device ? 'gpu' : 'cpu')
        : requestedBackend;
      if (resolvedBackend !== 'gpu') {
        return runCpu();
      }
      if (!device) {
        log.warn('Energy', 'Quintel GPU backend requested, but no WebGPU device is available.');
        return runCpu();
      }

      try {
        const caps = getKernelCapabilities();
        const wantsF16 = runtimeConfig.state?.dtype === 'f16';
        const dtype = wantsF16 && caps.hasF16 ? 'f16' : 'f32';
        if (wantsF16 && !caps.hasF16) {
          log.warn('Energy', 'Requested f16 state but device lacks f16 support. Using f32.');
        }
        log.info('Energy', 'Quintel backend: GPU');

        const statePayload = dtype === 'f16' ? f32ToF16Array(initData) : initData;
        const stateTensor = await createEnergyTensor(device, statePayload, dtype, shape, 'energy_state');

        const rules = quintelConfig.rules || {};
        const weights = quintelConfig.weights || {};
        const clampMin = Number.isFinite(quintelConfig.clamp?.min) ? quintelConfig.clamp.min : 0;
        const clampMax = Number.isFinite(quintelConfig.clamp?.max) ? quintelConfig.clamp.max : 1;
        const symmetryWeight = Number.isFinite(weights.symmetry) ? weights.symmetry : 1.0;
        const countWeight = Number.isFinite(weights.count) ? weights.count : 1.0;
        const centerWeight = Number.isFinite(weights.center) ? weights.center : 1.0;
        const binarizeWeight = Number.isFinite(weights.binarize) ? weights.binarize : 0.0;
        const centerTarget = Number.isFinite(quintelConfig.centerTarget) ? quintelConfig.centerTarget : 1.0;
        const energyHistory = [];
        const stepTimesMs = [];
        let lastEnergy = null;
        let lastComponents = null;
        let lastCountDiff = 0.0;

        const traceInterval = Math.max(0, Math.floor(traceEvery ?? diagnosticsConfig.traceEvery ?? 0));
        const reduceInterval = hasCountRule ? 1 : Math.max(1, Math.floor(readbackEvery));
        const reduceWorkgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
        const reduceBytes = reduceWorkgroups * 16;
        const reduceBuffer = acquireBuffer(reduceBytes, undefined, 'energy_quintel_reduce_output');
        const countTarget = Number.isFinite(quintelConfig.countTarget)
          ? quintelConfig.countTarget
          : size * size * 0.5;
        const hasSymmetryRules = !!rules.mirrorX || !!rules.mirrorY || !!rules.diagonal;
        const hasCountRule = !!rules.count;
        const hasCenterRule = !!rules.center;
        const hasBinarizeRule = Number.isFinite(binarizeWeight) && binarizeWeight > 0;
        const start = performance.now();

        for (let step = 0; step < maxSteps; step++) {
          const stepStart = performance.now();
          const shouldReduce = step % reduceInterval === 0 || step === maxSteps - 1;
          const shouldRecord = step % readbackEvery === 0 || step === maxSteps - 1;
          if (shouldReduce) {
            await runEnergyQuintelReduce(stateTensor, {
              count: elementCount,
              size,
              symmetryWeight,
              centerWeight,
              binarizeWeight,
              centerTarget,
              rules,
              outputBuffer: reduceBuffer,
            });

            const reduceData = await readBuffer(reduceBuffer, reduceBytes);
            const reduceValues = new Float32Array(reduceData);
            let sumState = 0.0;
            let symmetryEnergy = 0.0;
            let binarizeEnergy = 0.0;
            let centerEnergy = 0.0;
            for (let i = 0; i < reduceValues.length; i += 4) {
              sumState += reduceValues[i];
              symmetryEnergy += reduceValues[i + 1];
              binarizeEnergy += reduceValues[i + 2];
              centerEnergy += reduceValues[i + 3];
            }

            const countDiff = hasCountRule ? sumState - countTarget : 0.0;
            lastCountDiff = countDiff;

            const components = {
              symmetry: hasSymmetryRules ? symmetryEnergy : null,
              count: hasCountRule ? countWeight * countDiff * countDiff : null,
              center: hasCenterRule ? centerEnergy : null,
              binarize: hasBinarizeRule ? binarizeEnergy : null,
            };
            const energy = (components.symmetry ?? 0)
              + (components.count ?? 0)
              + (components.center ?? 0)
              + (components.binarize ?? 0);

            lastEnergy = energy;
            lastComponents = components;

            if (shouldRecord) {
              energyHistory.push(energy);
              if (energyHistory.length > historyLimit) {
                energyHistory.shift();
              }
            }

            if (traceInterval > 0 && traceEvery > 0 && step % traceInterval === 0) {
              trace.energy(`step=${step} energy=${energy.toFixed(6)}`, components);
            }

            if (step >= minSteps && convergenceThreshold != null && energy <= convergenceThreshold) {
              stepTimesMs.push(performance.now() - stepStart);
              break;
            }
          }

          const safeCountDiff = Number.isFinite(lastCountDiff) ? lastCountDiff : 0.0;
          await runEnergyQuintelUpdate(stateTensor, {
            count: elementCount,
            size,
            stepSize,
            gradientScale,
            countDiff: safeCountDiff,
            symmetryWeight,
            countWeight,
            centerWeight,
            binarizeWeight,
            centerTarget,
            clampMin,
            clampMax,
            rules,
          });

          stepTimesMs.push(performance.now() - stepStart);

          if (this._onProgress) {
            this._onProgress({
              stage: 'energy',
              percent: (step + 1) / maxSteps,
              message: `Step ${step + 1} / ${maxSteps}`,
            });
          }
        }

        const totalTimeMs = performance.now() - start;
        const finalState = await readTensorToFloat32(stateTensor);
        const stateStats = computeArrayStats(finalState);

        releaseBuffer(stateTensor.buffer);
        releaseBuffer(reduceBuffer);

        this.stats = {
          backend: 'GPU',
          totalTimeMs,
          steps: stepTimesMs.length,
          stepTimesMs,
          energyHistory,
          readbackCount: energyHistory.length,
          energy: lastEnergy,
          energyComponents: lastComponents,
          stateStats,
        };

        return {
          backend: 'GPU',
          shape,
          dtype,
          steps: stepTimesMs.length,
          energy: lastEnergy,
          state: finalState,
          energyHistory,
          energyComponents: lastComponents,
          stateStats,
          totalTimeMs,
          problem: 'quintel',
        };
      } catch (error) {
        log.warn('Energy', `GPU quintel path failed: ${error?.message || error}`);
        return runCpu();
      }
    }

    if (problem === 'vliw') {
      const vliwConfig = request.vliw ?? {};
      const tasks = Array.isArray(vliwConfig.tasks) ? vliwConfig.tasks : [];
      const caps = vliwConfig.caps ?? {};
      if (!tasks.length) {
        throw new Error('VLIW demo requires tasks in the request.');
      }
      const loop = {
        maxSteps,
        minSteps,
        stepSize,
        gradientScale,
        convergenceThreshold,
      };
      const diagnosticsConfig = {
        readbackEvery,
        historyLimit,
        traceEvery,
      };
      const result = runVliwEnergyLoop({
        tasks,
        caps,
        dependencyModel: vliwConfig.dependencyModel,
        loop,
        search: vliwConfig.search,
        seed: request.seed ?? runtimeConfig.init.seed,
        initMode: request.initMode ?? runtimeConfig.init.mode,
        initScale: request.initScale ?? runtimeConfig.init.scale,
        diagnostics: diagnosticsConfig,
        onProgress: this._onProgress,
        onTrace: (step, energy, metrics) => {
          if (!Number.isFinite(energy)) return;
          trace.energy(`step=${step} cycles=${energy}`, metrics);
        },
      });

      const energyComponents = {
        symmetry: result.metrics?.cycles ?? null,
        count: result.metrics?.utilization ?? null,
        binarize: result.metrics?.violations ?? null,
      };

      this.stats = {
        backend: 'CPU',
        totalTimeMs: result.totalTimeMs,
        steps: result.steps,
        stepTimesMs: null,
        energyHistory: result.energyHistory,
        readbackCount: result.energyHistory.length,
        energy: result.energy,
        energyComponents,
        stateStats: computeArrayStats(result.state),
      };

      return {
        backend: 'CPU',
        shape: result.shape,
        dtype: 'f32',
        steps: result.steps,
        energy: result.energy,
        state: result.state,
        energyHistory: result.energyHistory,
        energyComponents,
        stateStats: computeArrayStats(result.state),
        totalTimeMs: result.totalTimeMs,
        metrics: result.metrics,
        baseline: result.baseline,
        schedule: result.schedule,
        candidates: result.candidates,
        taskMeta: result.taskMeta,
        scheduler: result.scheduler,
        schedulerPolicy: result.schedulerPolicy,
        schedulerPolicies: result.schedulerPolicies,
        scoreMode: result.scoreMode,
        engineOrder: result.engineOrder,
        capsSource: result.capsSource,
        mode: result.mode,
        problem,
      };
    }

    const device = getDevice();
    if (!device) {
      throw new Error('Energy pipeline requires a WebGPU device.');
    }

    if (Array.isArray(request.shape) && request.shape.length > 0) {
      runtimeConfig.shape = request.shape.map((value) => Math.max(1, Math.floor(value)));
    } else {
      const width = Number.isFinite(request.width) ? Math.max(1, Math.floor(request.width)) : null;
      const height = Number.isFinite(request.height) ? Math.max(1, Math.floor(request.height)) : null;
      const channels = Number.isFinite(request.channels) ? Math.max(1, Math.floor(request.channels)) : null;
      if (width != null || height != null || channels != null) {
        runtimeConfig.state = { ...runtimeConfig.state };
        if (width != null) runtimeConfig.state.width = width;
        if (height != null) runtimeConfig.state.height = height;
        if (channels != null) runtimeConfig.state.channels = channels;
      }
    }
    const shape = resolveShape(runtimeConfig);
    const elementCount = shape.reduce((acc, value) => acc * value, 1);

    const caps = getKernelCapabilities();
    const wantsF16 = runtimeConfig.state.dtype === 'f16';
    const dtype = wantsF16 && caps.hasF16 ? 'f16' : 'f32';
    if (wantsF16 && !caps.hasF16) {
      log.warn('Energy', 'Requested f16 state but device lacks f16 support. Using f32.');
    }

    const initMode = request.initMode ?? runtimeConfig.init.mode;
    const initSeed = request.seed ?? runtimeConfig.init.seed;
    const initScale = request.initScale ?? runtimeConfig.init.scale;
    const targetMode = request.targetMode ?? runtimeConfig.target.mode;
    const targetSeed = request.targetSeed ?? runtimeConfig.target.seed;
    const targetScale = request.targetScale ?? runtimeConfig.target.scale;

    const initData = generateRandomArray(elementCount, initMode, initSeed, initScale);
    const targetData = generateRandomArray(elementCount, targetMode, targetSeed, targetScale);

    const statePayload = dtype === 'f16' ? f32ToF16Array(initData) : initData;
    const targetPayload = dtype === 'f16' ? f32ToF16Array(targetData) : targetData;

    const stateTensor = await createEnergyTensor(device, statePayload, dtype, shape, 'energy_state');
    const targetTensor = await createEnergyTensor(device, targetPayload, dtype, shape, 'energy_target');

    const energyHistory = [];
    const stepTimes = [];
    let readbackCount = 0;
    let lastEnergy = null;

    const energyBuffer = acquireBuffer(elementCount * 4, undefined, 'energy_eval_buffer');

    if (this._onProgress) {
      this._onProgress({ stage: 'energy', percent: 0, message: 'Initializing energy state' });
    }

    const start = performance.now();

    for (let step = 0; step < maxSteps; step++) {
      const stepStart = performance.now();
      await runEnergyUpdate(stateTensor, targetTensor, {
        count: elementCount,
        stepSize,
        gradientScale,
      });

      const shouldRead = step % readbackEvery === 0 || step === maxSteps - 1;
      if (shouldRead) {
        const energyTensor = await runEnergyEval(stateTensor, targetTensor, {
          count: elementCount,
          outputBuffer: energyBuffer,
        });
        const energyData = await readBuffer(energyTensor.buffer, elementCount * 4);
        const energyValues = new Float32Array(energyData);
        const stats = computeArrayStats(energyValues);
        lastEnergy = stats.mean;
        energyHistory.push(lastEnergy);
        readbackCount += 1;

        if (energyHistory.length > historyLimit) {
          energyHistory.shift();
        }

        if (traceEvery > 0 && step % traceEvery === 0) {
          trace.energy(`step=${step} meanEnergy=${lastEnergy.toFixed(6)}`, stats);
        }

        if (step >= minSteps && Number.isFinite(convergenceThreshold) && lastEnergy <= convergenceThreshold) {
          stepTimes.push(performance.now() - stepStart);
          break;
        }
      }

      stepTimes.push(performance.now() - stepStart);

      if (this._onProgress) {
        this._onProgress({
          stage: 'energy',
          percent: (step + 1) / maxSteps,
          message: `Step ${step + 1} / ${maxSteps}`,
        });
      }
    }

    const totalTimeMs = performance.now() - start;
    const stateData = await readTensorToFloat32(stateTensor);
    const stateStats = computeArrayStats(stateData);

    releaseBuffer(stateTensor.buffer);
    releaseBuffer(targetTensor.buffer);
    releaseBuffer(energyBuffer);

    this.stats = {
      backend: 'GPU',
      totalTimeMs,
      steps: stepTimes.length,
      stepTimesMs: stepTimes,
      energyHistory,
      readbackCount,
      energy: lastEnergy,
      stateStats,
    };

    return {
      backend: 'GPU',
      shape,
      dtype,
      steps: stepTimes.length,
      energy: lastEnergy,
      state: stateData,
      energyHistory,
      stateStats,
      totalTimeMs,
    };
  }
}

export async function createEnergyPipeline(manifest, contexts = {}) {
  const pipeline = new EnergyPipeline();
  await pipeline.initialize(contexts);
  await pipeline.loadModel(manifest);
  return pipeline;
}

registerPipeline('energy', createEnergyPipeline);
