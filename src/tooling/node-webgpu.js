import { existsSync, readFileSync, statSync } from 'node:fs';
import { dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const DOPPLER_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

const DEFAULT_LOCAL_DOE_PROVIDER_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '..',
  '..',
  '..',
  'fawn',
  'nursery',
  'webgpu-core',
);

function hasNavigatorGpu() {
  return typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator?.gpu;
}

function hasGpuEnums() {
  return typeof globalThis.GPUBufferUsage !== 'undefined' && typeof globalThis.GPUShaderStage !== 'undefined';
}

function setGlobalIfMissing(name, value) {
  if (value === undefined || value === null) return;
  if (globalThis[name] !== undefined) return;
  Object.defineProperty(globalThis, name, {
    value,
    writable: true,
    configurable: true,
    enumerable: false,
  });
}

function installGlobalsFromModule(mod) {
  const globals = mod?.globals;
  if (!globals || typeof globals !== 'object') return;
  for (const [name, value] of Object.entries(globals)) {
    setGlobalIfMissing(name, value);
  }
}

function resolveCandidateModuleSpecifier(candidate) {
  if (candidate.startsWith('file://')) {
    return candidate;
  }
  if (candidate.startsWith('.') || candidate.startsWith('/') || candidate.includes('/')) {
    const normalizedPath = isAbsolute(candidate)
      ? candidate
      : resolve(process.cwd(), candidate);
    const resolvedFilePath = resolveNodeModuleFilePath(normalizedPath);
    if (resolvedFilePath) {
      return pathToFileURL(resolvedFilePath).href;
    }
  }
  return candidate;
}

function resolveDefaultWebgpuModuleSpecifiers() {
  const specifiers = [];
  const localCandidates = [
    resolve(process.cwd(), '..', 'fawn', 'nursery', 'webgpu-core'),
    DEFAULT_LOCAL_DOE_PROVIDER_PATH,
  ];
  for (const localCandidate of localCandidates) {
    const resolvedPath = resolveNodeModuleFilePath(localCandidate);
    if (resolvedPath) {
      specifiers.push(pathToFileURL(resolvedPath).href);
    }
  }
  specifiers.push('@doe/webgpu-core');
  specifiers.push('webgpu');
  return [...new Set(specifiers)];
}

function resolveWebgpuModuleSpecifiers() {
  const fromEnv = process.env.DOPPLER_NODE_WEBGPU_MODULE;
  if (typeof fromEnv === 'string' && fromEnv.trim().length > 0) {
    return {
      explicit: true,
      specifiers: [resolveCandidateModuleSpecifier(fromEnv.trim())],
    };
  }
  return {
    explicit: false,
    specifiers: resolveDefaultWebgpuModuleSpecifiers(),
  };
}

function resolveWorkspaceWebgpuProviderPath() {
  const candidates = [
    resolve(process.cwd(), 'node_modules', 'webgpu'),
    resolve(DOPPLER_ROOT, 'node_modules', 'webgpu'),
  ];
  for (const candidate of candidates) {
    const resolvedPath = resolveNodeModuleFilePath(candidate);
    if (resolvedPath) {
      return resolvedPath;
    }
  }
  return null;
}

function isDoeWebgpuCoreSpecifier(specifier) {
  if (specifier === '@doe/webgpu-core') {
    return true;
  }
  if (typeof specifier !== 'string') {
    return false;
  }
  if (specifier.includes('/webgpu-core/')) {
    return true;
  }
  return specifier.includes('webgpu-core') && specifier.startsWith('file://');
}

function resolveDoeProviderOverride(specifier) {
  const explicitProvider = process.env.FAWN_WEBGPU_NODE_PROVIDER_MODULE;
  if (typeof explicitProvider === 'string' && explicitProvider.trim().length > 0) {
    return null;
  }
  if (!isDoeWebgpuCoreSpecifier(specifier)) {
    return null;
  }
  return resolveWorkspaceWebgpuProviderPath();
}

async function importWithProviderOverride(specifier) {
  const providerOverride = resolveDoeProviderOverride(specifier);
  if (!providerOverride) {
    return import(specifier);
  }
  const hadProvider = Object.prototype.hasOwnProperty.call(process.env, 'FAWN_WEBGPU_NODE_PROVIDER_MODULE');
  const previousProvider = process.env.FAWN_WEBGPU_NODE_PROVIDER_MODULE;
  process.env.FAWN_WEBGPU_NODE_PROVIDER_MODULE = providerOverride;
  try {
    return await import(specifier);
  } finally {
    if (hadProvider) {
      process.env.FAWN_WEBGPU_NODE_PROVIDER_MODULE = previousProvider;
    } else {
      delete process.env.FAWN_WEBGPU_NODE_PROVIDER_MODULE;
    }
  }
}

function resolveNodeModuleFilePath(candidatePath) {
  if (!existsSync(candidatePath)) return null;
  const stat = statSync(candidatePath);
  if (stat.isFile()) {
    return candidatePath;
  }
  if (!stat.isDirectory()) {
    return null;
  }
  const packageJsonPath = resolve(candidatePath, 'package.json');
  if (existsSync(packageJsonPath)) {
    try {
      const pkg = JSON.parse(readFileSync(packageJsonPath, 'utf8'));
      if (typeof pkg.main === 'string' && pkg.main.trim()) {
        const mainPath = resolve(candidatePath, pkg.main);
        if (existsSync(mainPath)) {
          return mainPath;
        }
      }
      const nodeExportPath = resolveExportsPath(pkg.exports, candidatePath);
      if (nodeExportPath) {
        return nodeExportPath;
      }
    } catch {
      // Ignore malformed package.json and fall through to file candidates.
    }
  }
  const fallbackPaths = [
    resolve(candidatePath, 'index.js'),
    resolve(candidatePath, 'src/index.js'),
    resolve(candidatePath, 'src/node-runtime.js'),
  ];
  for (const fallbackPath of fallbackPaths) {
    if (existsSync(fallbackPath)) {
      return fallbackPath;
    }
  }
  return null;
}

function resolveExportsPath(exportsField, rootPath) {
  if (!exportsField) return null;
  if (typeof exportsField === 'string') {
    const candidate = resolve(rootPath, exportsField);
    return existsSync(candidate) ? candidate : null;
  }
  if (typeof exportsField !== 'object' || Array.isArray(exportsField)) {
    return null;
  }
  if (typeof exportsField['./node'] === 'string') {
    const nodePath = resolve(rootPath, exportsField['./node']);
    if (existsSync(nodePath)) {
      return nodePath;
    }
  }
  const dot = exportsField['.'];
  if (typeof dot === 'string') {
    const dotPath = resolve(rootPath, dot);
    if (existsSync(dotPath)) {
      return dotPath;
    }
  } else if (dot && typeof dot === 'object') {
    const preferred = dot.default || dot.node || dot.import;
    if (typeof preferred === 'string') {
      const preferredPath = resolve(rootPath, preferred);
      if (existsSync(preferredPath)) {
        return preferredPath;
      }
    }
  }
  return null;
}

function installNavigatorGpu(gpu) {
  if (!gpu || typeof gpu.requestAdapter !== 'function') return false;
  if (typeof globalThis.navigator === 'undefined') {
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu },
      writable: true,
      configurable: true,
      enumerable: false,
    });
    return true;
  }

  if (!globalThis.navigator.gpu) {
    Object.defineProperty(globalThis.navigator, 'gpu', {
      value: gpu,
      writable: true,
      configurable: true,
      enumerable: false,
    });
  }
  return true;
}

function resolveGpuFromModule(mod) {
  if (!mod) return null;

  const fromModule = mod.gpu || mod.webgpu || mod.default?.gpu || mod.default?.webgpu;
  if (fromModule && typeof fromModule.requestAdapter === 'function') {
    return fromModule;
  }

  const factory = mod.create || mod.default?.create;
  if (typeof factory === 'function') {
    let created = null;
    try {
      created = factory([]);
    } catch {
      try {
        created = factory();
      } catch {
        created = null;
      }
    }
    if (created) {
      if (typeof created.requestAdapter === 'function') {
        return created;
      }
      if (created.gpu && typeof created.gpu.requestAdapter === 'function') {
        return created.gpu;
      }
    }
  }

  if (mod.default && typeof mod.default.requestAdapter === 'function') {
    return mod.default;
  }

  return null;
}

function installWebgpuFromModule(mod) {
  const gpu = resolveGpuFromModule(mod);
  if (!installNavigatorGpu(gpu)) {
    return false;
  }

  installGlobalsFromModule(mod);
  setGlobalIfMissing('GPUBufferUsage', mod.GPUBufferUsage || mod.default?.GPUBufferUsage || mod.globals?.GPUBufferUsage);
  setGlobalIfMissing('GPUShaderStage', mod.GPUShaderStage || mod.default?.GPUShaderStage || mod.globals?.GPUShaderStage);
  setGlobalIfMissing('GPUMapMode', mod.GPUMapMode || mod.default?.GPUMapMode || mod.globals?.GPUMapMode);
  setGlobalIfMissing('GPUTextureUsage', mod.GPUTextureUsage || mod.default?.GPUTextureUsage || mod.globals?.GPUTextureUsage);

  return hasNavigatorGpu() && hasGpuEnums();
}

export async function bootstrapNodeWebGPU() {
  if (hasNavigatorGpu() && hasGpuEnums()) {
    return true;
  }

  const { specifiers, explicit } = resolveWebgpuModuleSpecifiers();
  for (const specifier of specifiers) {
    let mod;
    try {
      mod = await importWithProviderOverride(specifier);
    } catch {
      if (explicit) {
        return false;
      }
      continue;
    }
    if (installWebgpuFromModule(mod)) {
      return true;
    }
    if (explicit) {
      return false;
    }
  }

  return false;
}
