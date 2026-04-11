import { loadLoRAFromManifest, loadLoRAFromUrl } from '../../adapters/lora-loader.js';
import { log } from '../../debug/index.js';
import { getDopplerLoader } from '../../loader/doppler-loader.js';
import { fetchArrayBuffer, readOPFSFile, writeOPFSFile } from './storage.js';

function createLoRALoadOptions() {
  return {
    readOPFS: readOPFSFile,
    writeOPFS: writeOPFSFile,
    fetchUrl: fetchArrayBuffer,
  };
}

function getActiveLoRAName(pipeline) {
  const active = pipeline?.getActiveLoRA?.() || null;
  return active ? active.name : null;
}

export async function loadLoRAAdapterForPipeline(pipeline, adapter) {
  if (!pipeline) {
    throw new Error('No model loaded. Call load() first.');
  }

  const options = createLoRALoadOptions();
  let lora;
  if (typeof adapter === 'string') {
    lora = await loadLoRAFromUrl(adapter, options);
  } else if (adapter?.adapterType === 'lora' || adapter?.modelType === 'lora') {
    const loader = pipeline.dopplerLoader || getDopplerLoader();
    await loader.init();
    lora = await loader.loadLoRAWeights(adapter);
  } else {
    lora = await loadLoRAFromManifest(adapter, options);
  }

  pipeline.setLoRAAdapter(lora);
  log.info('doppler', `LoRA adapter loaded: ${lora.name}`);
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
    const { readFile } = await import('node:fs/promises');
    const raw = await readFile(path, 'utf8');
    const manifest = JSON.parse(raw);
    await loadLoRAAdapterForPipeline(pipeline, manifest);
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
