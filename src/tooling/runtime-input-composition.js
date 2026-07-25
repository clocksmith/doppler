import { mergeRuntimeValues } from '../config/runtime-merge.js';

export function resolveRuntimeFromConfig(config) {
  if (!config || typeof config !== 'object') return null;
  if (config.runtime && typeof config.runtime === 'object') return config.runtime;
  if (config.shared || config.loading || config.inference || config.emulation) return config;
  return null;
}

function requireLoader(loader, label) {
  if (typeof loader !== 'function') {
    throw new Error(`runtime input composition does not support ${label} on this surface.`);
  }
  return loader;
}

function normalizeLoadedDocument(loaded, label) {
  if (!loaded || typeof loaded !== 'object') {
    throw new Error(`${label} did not resolve to a config object.`);
  }
  const config = (
    loaded.config
    && typeof loaded.config === 'object'
    && !Array.isArray(loaded.config)
  )
    ? loaded.config
    : loaded;
  const runtime = (
    loaded.runtime
    && typeof loaded.runtime === 'object'
    && !Array.isArray(loaded.runtime)
  )
    ? loaded.runtime
    : resolveRuntimeFromConfig(config);
  if (!runtime) {
    throw new Error(`${label} is missing runtime fields.`);
  }
  return { config, runtime };
}

function appendDocument(state, kind, ref, loaded, label) {
  const document = normalizeLoadedDocument(loaded, label);
  return {
    runtime: mergeRuntimeValues(state.runtime, document.runtime),
    documents: [
      ...state.documents,
      {
        kind,
        ref,
        config: document.config,
        runtime: document.runtime,
      },
    ],
  };
}

export async function resolveOrderedRuntimeInputs(
  initialRuntime,
  inputs = {},
  handlers = {},
  options = {}
) {
  let state = {
    runtime: initialRuntime,
    documents: [],
  };

  if (Array.isArray(inputs.configChain) && inputs.configChain.length > 0) {
    const loadConfig = requireLoader(handlers.loadRuntimeConfigFromRef, 'configChain');
    for (const ref of inputs.configChain) {
      const loaded = await loadConfig(ref, options);
      state = appendDocument(state, 'configChain', ref, loaded, `Loaded runtime config "${ref}"`);
    }
  }

  if (inputs.runtimeProfile) {
    const loadProfile = requireLoader(handlers.loadRuntimeProfile, 'runtimeProfile');
    const loaded = await loadProfile(inputs.runtimeProfile, options);
    state = appendDocument(
      state,
      'runtimeProfile',
      inputs.runtimeProfile,
      loaded,
      `Runtime profile "${inputs.runtimeProfile}"`
    );
  }

  if (inputs.runtimeConfigUrl) {
    const loadConfigUrl = requireLoader(handlers.loadRuntimeConfigFromUrl, 'runtimeConfigUrl');
    const loaded = await loadConfigUrl(inputs.runtimeConfigUrl, options);
    state = appendDocument(
      state,
      'runtimeConfigUrl',
      inputs.runtimeConfigUrl,
      loaded,
      `Runtime config URL "${inputs.runtimeConfigUrl}"`
    );
  }

  if (inputs.runtimeConfig) {
    state = appendDocument(
      state,
      'runtimeConfig',
      'inline',
      inputs.runtimeConfig,
      'runtimeConfig'
    );
  }

  return {
    runtime: state.runtime,
    documents: Object.freeze(state.documents.map((document) => Object.freeze(document))),
  };
}

function requireRuntimeBridge(runtimeBridge) {
  if (!runtimeBridge?.setRuntimeConfig) {
    throw new Error('runtime bridge must provide setRuntimeConfig().');
  }
  if (typeof runtimeBridge.getRuntimeConfig !== 'function') {
    throw new Error('runtime bridge must provide getRuntimeConfig().');
  }
}

export async function applyOrderedRuntimeInputs(runtimeBridge, inputs = {}, handlers = {}, options = {}) {
  requireRuntimeBridge(runtimeBridge);
  const resolved = await resolveOrderedRuntimeInputs(
    runtimeBridge.getRuntimeConfig(),
    inputs,
    handlers,
    options
  );
  if (resolved.documents.length > 0) {
    runtimeBridge.setRuntimeConfig(resolved.runtime);
  }
  return resolved;
}
