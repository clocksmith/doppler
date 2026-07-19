import { log } from '../../debug/index.js';
import { getDopplerLoader } from '../../loader/doppler-loader.js';
import { fetchArrayBuffer, readOPFSFile, writeOPFSFile } from './storage.js';

let loraModulePromise = null;

const LORA_TARGET_WEIGHT_KEYS = Object.freeze({
  q_proj: 'qProj',
  k_proj: 'kProj',
  v_proj: 'vProj',
  o_proj: 'oProj',
  gate_proj: 'gate',
  up_proj: 'up',
  down_proj: 'down',
  gate_up_proj: 'gateUp',
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
  if (!(lora?.layers instanceof Map) || lora.layers.size === 0) {
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

export async function loadLoRAAdapterForPipeline(pipeline, adapter, loadOptions = {}) {
  if (!pipeline) {
    throw new Error('No model loaded. Call load() first.');
  }

  const options = createLoRALoadOptions(loadOptions);
  let lora;
  if (typeof adapter === 'string') {
    const { loadLoRAFromUrl } = await getExperimentalLoRAModule();
    lora = await loadLoRAFromUrl(adapter, options);
  } else if (adapter?.adapterType === 'lora' || adapter?.modelType === 'lora') {
    const loader = pipeline.dopplerLoader || getDopplerLoader();
    await loader.init();
    lora = await loader.loadLoRAWeights(adapter);
  } else {
    const { loadLoRAFromManifest } = await getExperimentalLoRAModule();
    lora = await loadLoRAFromManifest(adapter, options);
  }

  assertLoRABaseModelForPipeline(pipeline, lora);
  assertLoRATargetsForPipeline(pipeline, lora);
  pipeline.setLoRAAdapter(lora);
  log.info('doppler', `LoRA adapter loaded: ${lora.name}`);
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
  pipeline.setLoRAAdapter(null);
  log.info('doppler', 'LoRA adapter unloaded');
}

export function getActiveLoRAForPipeline(pipeline) {
  return getActiveLoRAName(pipeline);
}
