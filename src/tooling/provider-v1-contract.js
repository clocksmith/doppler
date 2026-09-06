// Keep the existing wire contract readable; its name does not select a GPU engine.
export const NODE_WEBGPU_PROVIDER_SCHEMA = 'doe.webgpu-provider/v1';
const GLOBAL_NAMES = ['GPUBufferUsage', 'GPUShaderStage', 'GPUMapMode', 'GPUTextureUsage'];

function fail(code, message, stage, receipt = null, cause) {
  const error = new Error(message, cause === undefined ? undefined : { cause });
  error.name = 'DopplerNodeWebGPUProviderError';
  Object.assign(error, { code, stage, receipt });
  return error;
}

function requireKeys(value, keys, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
    || Object.keys(value).some(key => !keys.includes(key))) {
    throw fail('DOPPLER_PROVIDER_CONFIG_INVALID', `Invalid ${label}.`, 'configuration');
  }
}

function requirePath(value) {
  if (typeof value !== 'string' || !value.length
    || value.split('.').some(key => !key || ['__proto__', 'prototype', 'constructor'].includes(key))) {
    throw fail('DOPPLER_PROVIDER_CONFIG_INVALID', 'Provider export paths must be explicit safe property paths.', 'configuration');
  }
}

function validateOptions(options) {
  requireKeys(options, ['providers', 'adapterOptions', 'globals'], 'provider options');
  if (!Array.isArray(options.providers) || !options.providers.length) {
    throw fail('DOPPLER_PROVIDER_CONFIG_INVALID', 'Declare at least one authorized provider.', 'configuration');
  }
  if (options.adapterOptions !== null) {
    requireKeys(options.adapterOptions, ['powerPreference', 'forceFallbackAdapter', 'featureLevel'], 'adapterOptions');
    if ((options.adapterOptions.powerPreference !== undefined
      && !['low-power', 'high-performance'].includes(options.adapterOptions.powerPreference))
      || (options.adapterOptions.forceFallbackAdapter !== undefined && typeof options.adapterOptions.forceFallbackAdapter !== 'boolean')
      || (options.adapterOptions.featureLevel !== undefined && !['core', 'compatibility'].includes(options.adapterOptions.featureLevel))) {
      throw fail('DOPPLER_PROVIDER_CONFIG_INVALID', 'Invalid adapter selection requirements.', 'configuration');
    }
  }
  requireKeys(options.globals, ['mode'], 'globals policy');
  if (!['none', 'install-missing', 'replace'].includes(options.globals.mode)) {
    throw fail('DOPPLER_PROVIDER_CONFIG_INVALID', 'Declare the global installation mode.', 'configuration');
  }
  const ids = new Set();
  for (const provider of options.providers) {
    requireKeys(provider, provider?.kind === 'global' ? ['id', 'kind'] : ['id', 'kind', 'module', 'gpu', 'globals'], 'provider');
    if (typeof provider.id !== 'string' || !provider.id.trim() || ids.has(provider.id)) {
      throw fail('DOPPLER_PROVIDER_CONFIG_INVALID', 'Provider IDs must be nonempty and unique.', 'configuration');
    }
    ids.add(provider.id);
    if (provider.kind === 'global') continue;
    if (provider.kind !== 'module' || typeof provider.module !== 'string' || !provider.module.trim()) {
      throw fail('DOPPLER_PROVIDER_CONFIG_INVALID', 'Expected a global or explicitly named module provider.', 'configuration');
    }
    requireKeys(provider.gpu, ['kind', 'path', 'args', 'resultPath'], 'GPU export');
    requirePath(provider.gpu.path);
    if (!['export', 'factory'].includes(provider.gpu.kind)
      || (provider.gpu.kind === 'factory' && !Array.isArray(provider.gpu.args))
      || (provider.gpu.kind === 'export' && provider.gpu.args !== undefined)) {
      throw fail('DOPPLER_PROVIDER_CONFIG_INVALID', 'Factory providers require explicit arguments; exports do not accept arguments.', 'configuration');
    }
    if (provider.gpu.resultPath != null) requirePath(provider.gpu.resultPath);
    requireKeys(provider.globals, GLOBAL_NAMES, 'global export paths');
    for (const name of GLOBAL_NAMES) requirePath(provider.globals[name]);
  }
}

function resolveExport(root, propertyPath) {
  let owner;
  let value = root;
  for (const key of propertyPath.split('.')) {
    owner = value;
    value = owner?.[key];
  }
  return { owner, value };
}

function installGlobals(gpu, bindings, mode, receipt) {
  receipt.globals.installed = [];
  receipt.globals.restored = false;
  const writes = [];
  const restore = () => {
    const failures = [];
    for (const entry of [...writes].reverse()) {
      try {
        const current = Object.getOwnPropertyDescriptor(entry.object, entry.key);
        if (current?.value !== entry.value || current?.get || current?.set) {
          throw new Error(`${entry.label} changed outside the provider session; not overwriting it.`);
        }
        if (entry.previous) Object.defineProperty(entry.object, entry.key, entry.previous);
        else if (!Reflect.deleteProperty(entry.object, entry.key)) throw new Error(`Cannot restore ${entry.label}.`);
        writes.splice(writes.indexOf(entry), 1);
      } catch (error) { failures.push(error); }
    }
    receipt.globals.restored = !failures.length;
    if (failures.length) throw new AggregateError(failures, 'Node WebGPU global restoration failed.');
  };
  const install = (object, key, value, label) => {
    if (object[key] === value) return;
    if (mode === 'install-missing' && object[key] !== undefined) {
      throw new Error(`${label} already belongs to another provider; select replace explicitly.`);
    }
    const previous = Object.getOwnPropertyDescriptor(object, key);
    Object.defineProperty(object, key, { value, configurable: true, writable: true, enumerable: true });
    writes.push({ object, key, value, label, previous });
    receipt.globals.installed.push(label);
  };
  try {
    if (mode !== 'none') {
      for (const name of GLOBAL_NAMES) install(globalThis, name, bindings[name], name);
      if (globalThis.navigator == null) install(globalThis, 'navigator', {}, 'navigator');
      install(globalThis.navigator, 'gpu', gpu, 'navigator.gpu');
    }
    return restore;
  } catch (cause) {
    try { restore(); } catch (cleanup) {
      throw fail('DOPPLER_PROVIDER_CLEANUP_FAILED', 'Provider installation and restoration failed.', 'globals.restore', receipt,
        new AggregateError([cause, cleanup]));
    }
    throw cause;
  }
}

export async function openNodeWebGPU(input) {
  validateOptions(input);
  const options = structuredClone(input);
  const receipt = {
    schema: 'doe.webgpu-provider-receipt/v1', contract: NODE_WEBGPU_PROVIDER_SCHEMA,
    implementation: 'doppler', providers: options.providers,
    providerOrder: options.providers.map(provider => provider.id), adapterOptions: options.adapterOptions,
    globals: { mode: options.globals.mode, installed: [], restored: false },
    attempts: [], selectedProviderId: null, ok: false,
  };
  let lastError;
  for (const provider of options.providers) {
    let stage = 'module.import';
    let module = null;
    let gpu = null;
    let adapter = null;
    try {
      const bindings = {};
      if (provider.kind === 'global') {
        gpu = globalThis.navigator?.gpu;
        for (const name of GLOBAL_NAMES) bindings[name] = globalThis[name];
      } else {
        module = await import(provider.module);
        stage = 'gpu.resolve';
        const selected = resolveExport(module, provider.gpu.path);
        if (provider.gpu.kind === 'factory' && typeof selected.value !== 'function') throw new Error('Selected GPU factory is not callable.');
        gpu = provider.gpu.kind === 'factory'
          ? await selected.value.apply(selected.owner, provider.gpu.args) : selected.value;
        if (provider.gpu.resultPath != null) gpu = resolveExport(gpu, provider.gpu.resultPath).value;
        for (const name of GLOBAL_NAMES) bindings[name] = resolveExport(module, provider.globals[name]).value;
      }
      stage = 'gpu.validate';
      if (typeof gpu?.requestAdapter !== 'function') throw new Error('Selected provider does not expose GPU.requestAdapter().');
      for (const name of GLOBAL_NAMES) {
        if (!bindings[name] || !['object', 'function'].includes(typeof bindings[name])) throw new Error(`Selected provider is missing ${name}.`);
      }
      stage = 'adapter.request';
      adapter = await gpu.requestAdapter(options.adapterOptions ?? undefined);
      if (typeof adapter?.requestDevice !== 'function') throw new Error('Selected provider returned no usable WebGPU adapter.');
      stage = 'globals.install';
      const restore = installGlobals(gpu, bindings, options.globals.mode, receipt);
      receipt.attempts.push({ providerId: provider.id, kind: provider.kind, module: provider.module ?? null,
        ok: true, stage: 'complete', code: null, detail: null });
      receipt.selectedProviderId = provider.id;
      receipt.ok = true;
      let closed = false;
      return {
        get gpu() { return gpu; }, get adapter() { return adapter; }, get module() { return module; }, receipt,
        async close() {
          if (closed) return;
          restore();
          // WebGPU owns devices; callers destroy theirs before releasing this host.
          // Dawn's Node instance is reference-owned, not an invented destroy() API.
          gpu = null; adapter = null; module = null;
          closed = true;
        },
      };
    } catch (cause) {
      receipt.attempts.push({ providerId: provider.id, kind: provider.kind, module: provider.module ?? null,
        ok: false, stage, code: cause.code ?? 'DOPPLER_PROVIDER_UNAVAILABLE', detail: cause.message });
      gpu = null; adapter = null; module = null;
      if (cause.code === 'DOPPLER_PROVIDER_CLEANUP_FAILED') throw cause;
      lastError = cause;
    }
  }
  throw fail('DOPPLER_PROVIDER_UNAVAILABLE',
    `No authorized Node WebGPU provider succeeded: ${receipt.attempts.map(attempt => `${attempt.providerId}: ${attempt.detail}`).join('; ')}`,
    receipt.attempts.at(-1).stage, receipt, lastError);
}
