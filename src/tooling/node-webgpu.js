import { existsSync, readFileSync, statSync } from 'node:fs';
import { dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const DEFAULT_DOE_PROVIDER_CREATE_ARGS = 'enable-dawn-features=allow_unsafe_apis';
const DOE_PROVIDER_CREATE_ARGS_ENV = 'FAWN_WEBGPU_CREATE_ARGS';
const DOE_PROVIDER_SPECIFIERS = Object.freeze([
  '@doe-gpu/webgpu',
  '@simulatte/webgpu',
]);
const ADAPTER_PROBE_OPTIONS = Object.freeze([
  { powerPreference: 'high-performance' },
  { powerPreference: 'low-power' },
  null,
]);

function hasNavigatorGpu() {
  return typeof globalThis.navigator !== 'undefined'
    && !!globalThis.navigator?.gpu
    && typeof globalThis.navigator.gpu.requestAdapter === 'function';
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
  return [...DOE_PROVIDER_SPECIFIERS, 'webgpu'];
}

function resolveExplicitWebgpuModuleSpecifier() {
  const fromEnv = process.env.DOPPLER_NODE_WEBGPU_MODULE;
  if (typeof fromEnv === 'string' && fromEnv.trim().length > 0) {
    return resolveCandidateModuleSpecifier(fromEnv.trim());
  }
  return null;
}

function isDoeWebgpuSpecifier(specifier) {
  if (DOE_PROVIDER_SPECIFIERS.includes(specifier)) {
    return true;
  }
  return typeof specifier === 'string'
    && specifier.startsWith('file://')
    && DOE_PROVIDER_SPECIFIERS.some((candidate) => specifier.includes(candidate));
}

async function importWithProviderOverride(specifier) {
  const shouldApplyCreateArgsDefault = isDoeWebgpuSpecifier(specifier)
    && !(typeof process.env[DOE_PROVIDER_CREATE_ARGS_ENV] === 'string' && process.env[DOE_PROVIDER_CREATE_ARGS_ENV].trim().length > 0);
  if (!shouldApplyCreateArgsDefault) {
    return import(specifier);
  }
  process.env[DOE_PROVIDER_CREATE_ARGS_ENV] = DEFAULT_DOE_PROVIDER_CREATE_ARGS;
  try {
    return await import(specifier);
  } finally {
    delete process.env[DOE_PROVIDER_CREATE_ARGS_ENV];
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

function installNavigatorGpu(gpu, options = {}) {
  if (!gpu || typeof gpu.requestAdapter !== 'function') return false;
  const force = options.force === true;
  if (typeof globalThis.navigator === 'undefined') {
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu },
      writable: true,
      configurable: true,
      enumerable: false,
    });
    return true;
  }

  if (force && globalThis.navigator.gpu) {
    globalThis.navigator.gpu = gpu;
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

  const defaultCreateArgs = DEFAULT_DOE_PROVIDER_CREATE_ARGS.split(',').filter(Boolean);

  const tryCreateFactory = (factory) => {
    if (typeof factory !== 'function') {
      return null;
    }
    try {
      return factory(defaultCreateArgs);
    } catch {
      try {
        return factory([]);
      } catch {
        try {
          return factory();
        } catch {
          return null;
        }
      }
    }
  };

  const instanceFactory = mod.createInstance || mod.default?.createInstance;
  const createdFromInstanceFactory = tryCreateFactory(instanceFactory);
  if (createdFromInstanceFactory) {
    if (typeof createdFromInstanceFactory.requestAdapter === 'function') {
      return createdFromInstanceFactory;
    }
    if (createdFromInstanceFactory.gpu && typeof createdFromInstanceFactory.gpu.requestAdapter === 'function') {
      return createdFromInstanceFactory.gpu;
    }
  }

  const factory = mod.create || mod.default?.create;
  if (typeof factory === 'function') {
    const created = tryCreateFactory(factory);
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

function installWebgpuFromModule(mod, options = {}) {
  const gpu = resolveGpuFromModule(mod);
  if (!installNavigatorGpu(gpu, options)) {
    return false;
  }

  installGlobalsFromModule(mod);
  setGlobalIfMissing('GPUBufferUsage', mod.GPUBufferUsage || mod.default?.GPUBufferUsage || mod.globals?.GPUBufferUsage);
  setGlobalIfMissing('GPUShaderStage', mod.GPUShaderStage || mod.default?.GPUShaderStage || mod.globals?.GPUShaderStage);
  setGlobalIfMissing('GPUMapMode', mod.GPUMapMode || mod.default?.GPUMapMode || mod.globals?.GPUMapMode);
  setGlobalIfMissing('GPUTextureUsage', mod.GPUTextureUsage || mod.default?.GPUTextureUsage || mod.globals?.GPUTextureUsage);

  return hasNavigatorGpu() && hasGpuEnums();
}

function formatAdapterProbeDetail(error) {
  if (!error) {
    return 'requestAdapter returned null for high-performance, low-power, and default options.';
  }
  const message = error?.message || String(error);
  return `requestAdapter failed: ${message}`;
}

async function probeInstalledGpuAdapter() {
  if (!hasNavigatorGpu()) {
    return {
      ok: false,
      detail: 'navigator.gpu.requestAdapter is unavailable after provider installation.',
    };
  }
  if (!hasGpuEnums()) {
    return {
      ok: false,
      detail: 'WebGPU enums are unavailable after provider installation.',
    };
  }

  let lastError = null;
  for (const adapterOptions of ADAPTER_PROBE_OPTIONS) {
    try {
      const adapter = adapterOptions
        ? await globalThis.navigator.gpu.requestAdapter(adapterOptions)
        : await globalThis.navigator.gpu.requestAdapter();
      if (adapter) {
        return { ok: true, detail: null };
      }
    } catch (error) {
      lastError = error;
    }
  }

  return {
    ok: false,
    detail: formatAdapterProbeDetail(lastError),
  };
}

async function tryInstallAndProbeProvider(providerSpecifier, options = {}) {
  const specifier = resolveCandidateModuleSpecifier(providerSpecifier);
  let mod;
  try {
    mod = await importWithProviderOverride(specifier);
  } catch (error) {
    return {
      ok: false,
      provider: providerSpecifier,
      detail: `import failed: ${error?.message || String(error)}`,
      module: null,
    };
  }

  if (!installWebgpuFromModule(mod, { force: options.force === true })) {
    return {
      ok: false,
      provider: providerSpecifier,
      detail: 'failed to install WebGPU globals.',
      module: mod,
    };
  }

  const probe = await probeInstalledGpuAdapter();
  if (!probe.ok) {
    return {
      ok: false,
      provider: providerSpecifier,
      detail: probe.detail,
      module: mod,
    };
  }

  return {
    ok: true,
    provider: providerSpecifier,
    detail: null,
    module: mod,
  };
}

export async function bootstrapNodeWebGPUProvider(providerSpecifier, options = {}) {
  const attempt = await tryInstallAndProbeProvider(providerSpecifier, { force: options.force === true });
  if (!attempt.ok) {
    throw new Error(
      `node command: failed to install WebGPU provider "${providerSpecifier}": ${attempt.detail}`
    );
  }
  return {
    ok: true,
    provider: providerSpecifier,
    module: attempt.module,
  };
}

/**
 * Bootstrap Node WebGPU by attempting providers in priority order:
 *
 * 1. **Environment override** (`DOPPLER_NODE_WEBGPU_MODULE` env var) — highest priority.
 *    When set, only this specifier is attempted. Supports npm package names,
 *    relative/absolute file paths, and `file://` URLs.
 *
 * 2. **Pre-installed** — if `navigator.gpu` and GPU enums are already present
 *    (e.g., a WebGPU-enabled Node build or prior bootstrap), no module is loaded.
 *
 * 3. **Default candidates** — tried in order:
 *    a. `'@doe-gpu/webgpu'` (current Doe package name)
 *    b. `'@simulatte/webgpu'` (legacy package name during transition)
 *    c. `'webgpu'` (community Dawn bindings)
 *    The first one that imports, installs, and passes an adapter probe wins.
 *
 * @returns {Promise<{ ok: boolean, provider: string | null }>}
 */
export async function bootstrapNodeWebGPU() {
  const explicitSpecifier = resolveExplicitWebgpuModuleSpecifier();
  if (explicitSpecifier) {
    const attempt = await tryInstallAndProbeProvider(explicitSpecifier, { force: true });
    return {
      ok: attempt.ok,
      provider: explicitSpecifier,
      detail: attempt.detail,
    };
  }

  if (hasNavigatorGpu() && hasGpuEnums()) {
    const preinstalledProbe = await probeInstalledGpuAdapter();
    if (preinstalledProbe.ok) {
      return { ok: true, provider: 'pre-installed', detail: null };
    }
  }

  let lastFailure = null;
  for (const specifier of resolveDefaultWebgpuModuleSpecifiers()) {
    const attempt = await tryInstallAndProbeProvider(specifier, { force: true });
    if (attempt.ok) {
      return { ok: true, provider: specifier, detail: null };
    }
    lastFailure = attempt;
  }

  return {
    ok: false,
    provider: lastFailure?.provider ?? null,
    detail: lastFailure?.detail ?? null,
  };
}
