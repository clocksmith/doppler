import { getDevice, setDevice } from '../../gpu/device.js';
import { log, applyDebugConfig, setGPUDevice } from '../../debug/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../config/runtime.js';
import { registerPipeline } from '../pipeline/registry.js';
import { initializeDiffusion } from './init.js';
import { loadDiffusionTokenizers, encodePrompt } from './text-encoder.js';
import { buildScheduler } from './scheduler.js';
import { runUnetStep } from './unet.js';
import { decodeLatents } from './vae.js';

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

function generateLatents(width, height, channels, latentScale, seed) {
  const latentWidth = Math.max(1, Math.floor(width / latentScale));
  const latentHeight = Math.max(1, Math.floor(height / latentScale));
  const size = latentWidth * latentHeight * channels;
  const latents = new Float32Array(size);
  const rand = createRng(seed ?? Math.floor(Math.random() * 1e9));
  for (let i = 0; i < size; i++) {
    const u = rand();
    const v = rand();
    const z = Math.sqrt(-2.0 * Math.log(Math.max(u, 1e-6))) * Math.cos(2.0 * Math.PI * v);
    latents[i] = z;
  }
  return { latents, latentWidth, latentHeight };
}

export class DiffusionPipeline {
  
  runtimeConfig = null;
  
  manifest = null;
  
  diffusionState = null;
  
  tokenizers = null;
  
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
    if (!manifest || manifest.modelType !== 'diffusion') {
      throw new Error('Diffusion pipeline requires a diffusion model manifest.');
    }
    this.manifest = manifest;
    this.diffusionState = initializeDiffusion(manifest, this.runtimeConfig);
    this.tokenizers = await loadDiffusionTokenizers(this.diffusionState.modelConfig, {
      baseUrl: this.baseUrl,
    });
    log.info('Diffusion', `Loaded diffusion model "${manifest.modelId}" with ${Object.keys(this.tokenizers || {}).length} tokenizers`);
    log.warn('Diffusion', 'Diffusion kernels are not implemented yet; using CPU placeholder pipeline.');
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
    this.tokenizers = null;
    this.manifest = null;
    this.diffusionState = null;
  }

  async generate(request = {}) {
    if (!this.diffusionState) {
      throw new Error('Diffusion pipeline not initialized.');
    }
    const start = performance.now();
    const runtime = this.diffusionState.runtime;
    const width = request.width ?? runtime.latent.width;
    const height = request.height ?? runtime.latent.height;
    const steps = request.steps ?? runtime.scheduler.numSteps;
    const guidanceScale = request.guidanceScale ?? runtime.scheduler.guidanceScale;
    const seed = request.seed ?? Math.floor(Math.random() * 1e9);

    const promptStart = performance.now();
    const encoded = encodePrompt(
      { prompt: request.prompt ?? '', negativePrompt: request.negativePrompt ?? '' },
      this.tokenizers || {},
      { maxLength: runtime.textEncoder.maxLength }
    );
    const promptEnd = performance.now();

    const scheduler = buildScheduler(runtime.scheduler, steps);
    const latentScale = this.diffusionState.latentScale;
    const latentChannels = this.diffusionState.latentChannels;
    const { latents, latentWidth, latentHeight } = generateLatents(width, height, latentChannels, latentScale, seed);

    this._onProgress?.({
      stage: 'diffusion',
      message: `Denoising ${scheduler.steps} steps...`,
      progress: 0,
    });

    const decodeStart = performance.now();
    for (let i = 0; i < scheduler.steps; i++) {
      runUnetStep(latents, scheduler, i, guidanceScale);
      if (i % 5 === 0 || i === scheduler.steps - 1) {
        this._onProgress?.({
          stage: 'diffusion',
          message: `Denoising ${i + 1}/${scheduler.steps}`,
          progress: (i + 1) / scheduler.steps,
        });
      }
    }
    const decodeEnd = performance.now();

    const pixels = decodeLatents(latents, {
      width,
      height,
      latentWidth,
      latentHeight,
      latentChannels,
      latentScale,
    });

    const end = performance.now();
    this.stats = {
      totalTimeMs: end - start,
      prefillTimeMs: promptEnd - promptStart,
      prefillTokens: encoded.totalTokens,
      decodeTimeMs: decodeEnd - decodeStart,
      decodeTokens: scheduler.steps,
    };

    return { width, height, pixels };
  }
}

export async function createDiffusionPipeline(manifest, contexts = {}) {
  const pipeline = new DiffusionPipeline();
  await pipeline.initialize(contexts);
  await pipeline.loadModel(manifest);
  return pipeline;
}

registerPipeline('diffusion', createDiffusionPipeline);
