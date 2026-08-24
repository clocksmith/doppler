import { parseModelConfigFromManifest } from '../pipelines/text/config.js';
import { activateKernelPathState, resolveKernelPathState } from '../pipelines/text/model-load.js';
import { parseManifest } from '../../formats/rdrr/index.js';
import { getKernelCapabilities } from '../../gpu/device.js';
import { loadManifestFromStore, openModelStore } from '../../storage/shard-manager.js';

const DIRECT_SOURCE_FILE_EXTENSIONS = Object.freeze([
  '.gguf',
  '.tflite',
  '.task',
  '.litertlm',
]);

function isNodeRuntime() {
  return typeof process !== 'undefined' && !!process.versions?.node;
}

async function pathExists(nodeFs, targetPath) {
  try {
    await nodeFs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

function isDirectSourceFilePath(nodePath, targetPath) {
  const ext = nodePath.extname(String(targetPath || '')).toLowerCase();
  return DIRECT_SOURCE_FILE_EXTENSIONS.includes(ext);
}

async function loadNodeFsHelpers() {
  const [{ default: fs }, { default: path }, { fileURLToPath }] = await Promise.all([
    import('node:fs/promises'),
    import('node:path'),
    import('node:url'),
  ]);
  return { fs, path, fileURLToPath };
}

export function resolveDeviceInfo() {
  try {
    return getKernelCapabilities();
  } catch {
    return null;
  }
}

export async function resolveKernelPathForModel(options = {}) {
  const runtimeConfig = options.runtime?.runtimeConfig ?? null;
  let manifest = null;
  let manifestModelId = options.modelId || null;

  if (options.modelId) {
    await openModelStore(options.modelId);
    const manifestText = await loadManifestFromStore();
    if (manifestText) {
      manifest = parseManifest(manifestText);
      manifestModelId = manifest.modelId ?? options.modelId;
    }
  }

  if (!manifest) return null;

  const modelConfig = parseModelConfigFromManifest(
    manifest,
    runtimeConfig?.inference?.modelOverrides ?? null
  );
  const kernelPathState = resolveKernelPathState({
    manifest,
    runtimeConfig,
    modelConfig,
  });
  activateKernelPathState(kernelPathState);
  return {
    modelId: manifestModelId,
    kernelPath: kernelPathState.resolvedKernelPath,
    source: kernelPathState.kernelPathSource,
  };
}

export async function resolveLocalSourceRuntimePathFromModelUrl(modelUrl) {
  if (!isNodeRuntime()) {
    return null;
  }
  if (typeof modelUrl !== 'string' || !modelUrl.startsWith('file://')) {
    return null;
  }

  const { fs, path, fileURLToPath } = await loadNodeFsHelpers();

  let localPath;
  try {
    localPath = fileURLToPath(modelUrl);
  } catch {
    return null;
  }

  let stats;
  try {
    stats = await fs.stat(localPath);
  } catch {
    return null;
  }

  if (stats.isFile()) {
    return isDirectSourceFilePath(path, localPath) ? localPath : null;
  }

  if (!stats.isDirectory()) {
    return null;
  }

  if (await pathExists(fs, path.join(localPath, 'manifest.json'))) {
    return null;
  }

  const entries = await fs.readdir(localPath, { withFileTypes: true });
  const fileNames = new Set(
    entries
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name)
  );
  const hasSafetensorsShape = fileNames.has('config.json')
    && (fileNames.has('model.safetensors') || fileNames.has('model.safetensors.index.json'));
  if (hasSafetensorsShape) {
    return localPath;
  }

  for (const fileName of fileNames) {
    if (isDirectSourceFilePath(path, fileName)) {
      return localPath;
    }
  }

  return null;
}
