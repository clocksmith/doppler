import { authorizeBundledAdapter } from '../../config/revocation-policy.js';
import { log } from '../../debug/index.js';
import { getDopplerLoader } from '../../loader/doppler-loader.js';
import { fetchArrayBuffer, readOPFSFile, writeOPFSFile } from './storage.js';
import { preparePipelineAdapter, scopePipelineShaders, updatePipelineAdapter } from '../../inference/pipelines/shader-scoped-pipeline.js';

let loraModulePromise = null;
const preparedAdapters = new WeakMap();

const LORA_TARGET_WEIGHT_KEYS = Object.freeze({
  q_proj: 'qProj',
  k_proj: 'kProj',
  v_proj: 'vProj',
  o_proj: 'oProj',
  gate_proj: 'gate',
  up_proj: 'up',
  down_proj: 'down',
  gate_up_proj: 'gateUp',
  in_proj_a: 'linearInProjA',
  in_proj_b: 'linearInProjB',
  in_proj_qkv: 'qkvProj',
  in_proj_z: 'linearInProjZ',
  out_proj: 'oProj',
});

async function getExperimentalLoRAModule() {
  loraModulePromise ??= import('../../experimental/adapters/lora-loader.js');
  return loraModulePromise;
}

function createLoRALoadOptions(overrides = {}) {
  return {
    readOPFS: readOPFSFile,
    writeOPFS: writeOPFSFile,
    fetchUrl: fetchArrayBuffer,
    ...overrides,
  };
}

function getActiveLoRAName(pipeline) {
  const active = pipeline?.getActiveLoRA?.() || null;
  return active ? active.name : null;
}

function getActiveLoRAIdentity(pipeline) {
  const active = pipeline?.getActiveLoRA?.() || null;
  return active ? getLoRAIdentity(active) : null;
}

function getLoRAIdentity(adapter) {
  const identity = adapter.identity;
  if (identity?.schema !== 'doppler.lora-execution-identity/v1'
    || !/^sha256:[a-f0-9]{64}$/.test(String(identity.digest || ''))) {
    throw new Error(
      'LoRA adapter lacks exact tensor identity. Reload it through the Doppler adapter loader.'
    );
  }
  return identity;
}

function getPipelineModelId(pipeline) {
  return String(
    pipeline?.manifest?.modelId ||
    pipeline?.dopplerLoader?.manifest?.modelId ||
    pipeline?.dopplerLoader?.modelId ||
    ''
  ).trim();
}

export function assertLoRABaseModelForPipeline(pipeline, lora) {
  const modelId = getPipelineModelId(pipeline);
  if (!modelId) {
    throw new Error('LoRA activation requires a loaded pipeline with an exact manifest modelId.');
  }
  const baseModel = String(lora?.baseModel || '').trim();
  if (!baseModel) {
    throw new Error('LoRA activation requires the adapter to declare baseModel.');
  }
  if (baseModel !== modelId) {
    throw new Error(
      `LoRA adapter targets base model "${baseModel}", but the loaded model is "${modelId}".`
    );
  }
}

export function assertLoRATargetsForPipeline(pipeline, lora) {
  if (!(pipeline?.weights instanceof Map)) {
    throw new Error('LoRA activation requires loaded per-layer model weights.');
  }
  if (!(lora?.layers instanceof Map) || lora.layers.size === 0 && !pipeline.modelPartition) {
    throw new Error('LoRA activation requires at least one loaded adapter layer.');
  }

  for (const [layerIndex, modules] of lora.layers.entries()) {
    const layerWeights = pipeline.weights.get(`layer_${layerIndex}`);
    if (!layerWeights) {
      throw new Error(
        `LoRA adapter targets layer ${layerIndex}, which is absent from the loaded model.`
      );
    }
    for (const moduleName of Object.keys(modules)) {
      const weightKey = LORA_TARGET_WEIGHT_KEYS[moduleName];
      if (!weightKey || !layerWeights[weightKey]) {
        throw new Error(
          `LoRA adapter targets ${moduleName} at layer ${layerIndex}, but the loaded model has no compatible weight.`
        );
      }
    }
  }
}

async function acquireLoRAAdapter(pipeline, adapter, loadOptions) {
    loadOptions.signal?.throwIfAborted();
    const options = createLoRALoadOptions(loadOptions);
    let lora;
    if (typeof adapter === 'string') {
      const { loadLoRAFromUrl } = await getExperimentalLoRAModule();
      lora = await loadLoRAFromUrl(adapter, options);
    } else if (adapter?.adapterType === 'lora' || adapter?.modelType === 'lora') {
      const loader = pipeline.dopplerLoader || getDopplerLoader();
      await loader.init();
      const loaded = await loader.loadLoRAWeights(adapter);
      lora = loaded?.adapter ?? loaded;
    } else {
      const { loadLoRAFromManifest } = await getExperimentalLoRAModule();
      lora = await loadLoRAFromManifest(adapter, options);
    }
    loadOptions.signal?.throwIfAborted();
    assertLoRABaseModelForPipeline(pipeline, lora);
    if (pipeline.modelPartition) {
      const { plan, index } = pipeline.modelPartition;
      if (JSON.stringify(plan.adapterIdentities) !== JSON.stringify([lora.identity?.digest])) {
        throw new Error('LoRA execution identity differs from the accepted partition plan.');
      }
      for (const layer of lora.layers.keys()) {
        if (!Number.isSafeInteger(layer) || layer < 0 || layer >= plan.totalLayers) {
          throw new Error('LoRA targets a layer outside the base model.');
        }
      }
      const [first, last] = plan.partitions[index].layerRange;
      // Verify the complete source before retaining only this stage's A/B pairs.
      // The global digest identifies that source; planId binds the local layer range.
      lora = await authorizeBundledAdapter({ ...lora,
        layers: new Map([...lora.layers].filter(([layer]) => layer >= first && layer <= last)) });
    }
    assertLoRATargetsForPipeline(pipeline, lora);
    loadOptions.signal?.throwIfAborted();
    return lora;
}

export async function loadLoRAAdapterForPipeline(pipeline, adapter, loadOptions = {}) {
  if (!pipeline) throw new Error('No model loaded. Call load() first.');
  const activated = await updatePipelineAdapter(pipeline, () => acquireLoRAAdapter(pipeline, adapter, loadOptions));
  log.info('doppler', `LoRA adapter loaded: ${activated.name}`);
}

export async function prepareLoRAAdapterForPipeline(pipeline, adapter, loadOptions = {}) {
  if (!pipeline) throw new Error('No model loaded. Call load() first.');
  const scoped = scopePipelineShaders(pipeline);
  return preparePipelineAdapter(scoped, async () => {
    const lora = await acquireLoRAAdapter(scoped, adapter, loadOptions);
    const prepared = Object.freeze({ schema: 'doppler.prepared-lora/v1', name: lora.name,
      baseModel: getPipelineModelId(scoped), identity: Object.freeze({ ...getLoRAIdentity(lora) }),
      get released() { return !preparedAdapters.has(prepared); },
      release() { preparedAdapters.delete(prepared); } });
    preparedAdapters.set(prepared, { pipeline: scoped, adapter: lora,
      manifest: scoped.manifest, execution: scoped.resolvedRuntimeSession });
    return prepared;
  });
}

export async function activatePreparedLoRAForPipeline(pipeline, prepared) {
  if (!pipeline) throw new Error('No model loaded. Call load() first.');
  const scoped = scopePipelineShaders(pipeline);
  await updatePipelineAdapter(scoped, () => {
    const entry = preparedAdapters.get(prepared);
    if (!entry || entry.pipeline !== scoped) throw new Error('Prepared LoRA is released, unknown or belongs to another model owner.');
    if (entry.manifest !== scoped.manifest || entry.execution !== scoped.resolvedRuntimeSession) {
      throw new Error('Prepared LoRA base execution changed; prepare it again for this model.');
    }
    if (scoped.currentSeqLen !== 0) throw new Error('Reset generation state before activating a prepared LoRA adapter.');
    const lora = entry.adapter;
    if (getLoRAIdentity(lora).digest !== prepared.identity.digest) throw new Error('Prepared LoRA identity changed.');
    assertLoRABaseModelForPipeline(scoped, lora);
    assertLoRATargetsForPipeline(scoped, lora);
    if (scoped.modelPartition && JSON.stringify(scoped.modelPartition.plan.adapterIdentities)
      !== JSON.stringify([lora.identity.digest])) throw new Error('Prepared LoRA differs from the accepted partition plan.');
    return lora;
  });
}

async function readLocalJson(path) {
  const { readFile } = await import('node:fs/promises');
  const raw = await readFile(path, 'utf8');
  return JSON.parse(raw);
}

async function createLocalFileLoadOptions(path) {
  const { readFile } = await import('node:fs/promises');
  const { dirname, isAbsolute, join } = await import('node:path');
  const basePath = dirname(path);
  return {
    basePath,
    resolvePath(filePath) {
      return isAbsolute(filePath) ? filePath : join(basePath, filePath);
    },
    async readFile(filePath) {
      const data = await readFile(filePath);
      return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    },
  };
}

export async function activateLoRAFromTrainingOutputForPipeline(pipeline, trainingOutput) {
  if (!pipeline) {
    return {
      activated: false,
      adapterName: null,
      source: null,
      reason: 'no_model_loaded',
    };
  }

  const output = trainingOutput && typeof trainingOutput === 'object'
    ? trainingOutput
    : null;
  if (!output && typeof trainingOutput !== 'string') {
    return {
      activated: false,
      adapterName: getActiveLoRAName(pipeline),
      source: null,
      reason: 'no_adapter_candidate',
    };
  }

  if (typeof trainingOutput === 'string') {
    await loadLoRAAdapterForPipeline(pipeline, trainingOutput);
    return {
      activated: true,
      adapterName: getActiveLoRAName(pipeline),
      source: 'adapter-string',
      reason: null,
    };
  }

  if (output.adapterManifest && typeof output.adapterManifest === 'object') {
    await loadLoRAAdapterForPipeline(pipeline, output.adapterManifest);
    return {
      activated: true,
      adapterName: getActiveLoRAName(pipeline),
      source: 'adapterManifest',
      reason: null,
    };
  }

  if (typeof output.adapterManifestJson === 'string' && output.adapterManifestJson.trim()) {
    const manifest = JSON.parse(output.adapterManifestJson);
    await loadLoRAAdapterForPipeline(pipeline, manifest);
    return {
      activated: true,
      adapterName: getActiveLoRAName(pipeline),
      source: 'adapterManifestJson',
      reason: null,
    };
  }

  if (typeof output.adapterManifestUrl === 'string' && output.adapterManifestUrl.trim()) {
    await loadLoRAAdapterForPipeline(pipeline, output.adapterManifestUrl.trim());
    return {
      activated: true,
      adapterName: getActiveLoRAName(pipeline),
      source: 'adapterManifestUrl',
      reason: null,
    };
  }

  if (typeof output.adapterManifestPath === 'string' && output.adapterManifestPath.trim()) {
    const path = output.adapterManifestPath.trim();
    if (path.startsWith('http://') || path.startsWith('https://')) {
      await loadLoRAAdapterForPipeline(pipeline, path);
      return {
        activated: true,
        adapterName: getActiveLoRAName(pipeline),
        source: 'adapterManifestPath:url',
        reason: null,
      };
    }

    const isNode = typeof process !== 'undefined' && !!process.versions?.node;
    if (!isNode) {
      throw new Error('adapterManifestPath local files require Node runtime.');
    }
    const manifest = await readLocalJson(path);
    await loadLoRAAdapterForPipeline(pipeline, manifest, await createLocalFileLoadOptions(path));
    return {
      activated: true,
      adapterName: getActiveLoRAName(pipeline),
      source: 'adapterManifestPath:file',
      reason: null,
    };
  }

  if (output.adapter != null) {
    await loadLoRAAdapterForPipeline(pipeline, output.adapter);
    return {
      activated: true,
      adapterName: getActiveLoRAName(pipeline),
      source: 'adapter',
      reason: null,
    };
  }

  return {
    activated: false,
    adapterName: getActiveLoRAName(pipeline),
    source: null,
    reason: 'no_adapter_candidate',
  };
}

export async function unloadLoRAAdapterForPipeline(pipeline) {
  if (!pipeline) return;
  await updatePipelineAdapter(pipeline, () => null);
  log.info('doppler', 'LoRA adapter unloaded');
}

export function getActiveLoRAForPipeline(pipeline) {
  return getActiveLoRAName(pipeline);
}

export function getActiveLoRAIdentityForPipeline(pipeline) {
  return getActiveLoRAIdentity(pipeline);
}
