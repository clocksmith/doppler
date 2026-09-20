import {
  runWithShaderSourceScope,
  streamWithShaderSourceScope,
  getStorageShaderSourceScope,
} from '../../gpu/kernels/shader-source-scope.js';
import { applyPipelineContexts } from './context.js';
import { isDeviceLost } from '../../gpu/device-state.js';

const owners = new WeakMap();
const synchronousMutations = new Set(['setLoRAAdapter', 'reset', 'resetGenerationState', 'resetToSeqLen']);

function assertOpen(owner) {
  if (owner.closing) throw new Error('Pipeline is closing or closed; reload the model before starting work.');
  if (owner.pipeline.isLoaded === false) throw new Error('Pipeline is unloaded; load the model before starting work.');
}

function assertMutable(owner) {
  assertOpen(owner);
  if (owner.pending || owner.mutating || owner.pipeline.isGenerating) {
    throw new Error('Pipeline operation is in progress; finish active work before changing session state.');
  }
}

function beginOperation(owner) {
  assertOpen(owner);
  if (owner.mutating) throw new Error('Pipeline adapter change is in progress; finish it before starting work.');
  owner.pending += 1;
  return () => { owner.pending -= 1; notifyIdle(owner); };
}

function notifyIdle(owner) {
  if (!owner.pending && !owner.mutating) owner.onIdle?.();
}

export async function runPipelineOperation(pipeline, action) {
  const owner = owners.get(scopePipelineShaders(pipeline));
  // Internal calls use the original pipeline, as scoped methods do, without
  // acquiring their own lease again. External handles retain the guarded view.
  return runOperation(owner, 'execute', () => action(owner.pipeline));
}

async function runOperation(owner, key, action) {
  const release = beginOperation(owner);
  try {
    return await runWithShaderSourceScope(owner.scope, () => invoke(owner.pipeline, key, () => {
      assertOpen(owner);
      return action();
    }));
  } finally { release(); }
}

async function* streamOperation(owner, key, action) {
  const release = beginOperation(owner);
  try {
    yield* streamWithShaderSourceScope(owner.scope, () => stream(owner.pipeline, key, () => {
      assertOpen(owner);
      return action();
    }));
  } finally { release(); }
}

function closeOwner(owner, action) {
  if (!owner.closeTask) {
    owner.closing = true;
    const idle = owner.pending || owner.mutating
      ? new Promise(resolve => { owner.onIdle = resolve; }) : Promise.resolve();
    owner.closeTask = idle.then(() => runWithShaderSourceScope(
      owner.scope, () => invoke(owner.pipeline, 'unload', action)
    ));
  }
  return owner.closeTask;
}

export async function updatePipelineAdapter(pipeline, prepare) {
  const scoped = scopePipelineShaders(pipeline);
  const owner = owners.get(scoped);
  assertMutable(owner);
  owner.mutating = true;
  try {
    return await runWithShaderSourceScope(owner.scope, () => invoke(owner.pipeline, 'setLoRAAdapter', async () => {
      assertOpen(owner);
      const adapter = await prepare();
      assertOpen(owner);
      assertDeviceAvailable(owner.pipeline, 'setLoRAAdapter');
      owner.pipeline.setLoRAAdapter(adapter);
      return adapter;
    }));
  } finally { owner.mutating = false; notifyIdle(owner); }
}

function assertDeviceAvailable(pipeline, operation) {
  const lost = isDeviceLost(pipeline.gpuContext?.device);
  if (lost && operation !== 'unload' && operation !== 'cleanup') {
    throw new Error('Pipeline device is lost; reload this model on a live device.');
  }
  return lost;
}

function enterCompatibilityContext(pipeline, operation) {
  const lost = assertDeviceAvailable(pipeline, operation);
  return applyPipelineContexts({}, {
    runtimeConfig: pipeline.runtimeConfig,
    gpu: lost ? null : pipeline.gpuContext,
  }).restore;
}

async function invoke(pipeline, operation, action) {
  const restore = enterCompatibilityContext(pipeline, operation);
  try { return await action(); } finally { restore(); }
}

async function* stream(pipeline, operation, action) {
  const restore = enterCompatibilityContext(pipeline, operation);
  try { yield* action(); } finally { restore(); }
}

export function scopePipelineShaders(pipeline, scope) {
  const existing = owners.get(pipeline);
  if (existing) {
    if (scope !== undefined && scope !== existing.scope) throw new Error('Pipeline shader scope is already bound.');
    return existing.proxy;
  }
  const owner = { pipeline, scope: scope === undefined ? getStorageShaderSourceScope(pipeline.storageContext) : scope,
    pending: 0, mutating: false, closing: false, closeTask: null, onIdle: null, proxy: null };
  const methods = new Map();
  const proxy = new Proxy(pipeline, {
    get(target, key) {
      const value = Reflect.get(target, key, target);
      if (typeof value !== 'function') return value;
      if (methods.get(key)?.original === value) return methods.get(key).bound;
      // Text pipelines retain synchronous forwarding methods for their
      // generation controller. Preserve the controller's async/stream shape.
      const kind = value.constructor.name === 'Function' && typeof target.generator?.[key] === 'function'
        ? target.generator[key].constructor.name : value.constructor.name;
      let bound;
      if (key === 'unload') {
        bound = (...args) => closeOwner(owner, () => value.apply(target, args));
      } else if (kind === 'AsyncGeneratorFunction') {
        bound = (...args) => streamOperation(owner, key, () => value.apply(target, args));
      } else if (kind === 'AsyncFunction') {
        bound = (...args) => runOperation(owner, key, () => value.apply(target, args));
      } else if (synchronousMutations.has(key)) {
        bound = (...args) => { assertMutable(owner); return value.apply(target, args); };
      } else {
        bound = value.bind(target);
      }
      methods.set(key, { original: value, bound });
      return bound;
    },
  });
  owner.proxy = proxy;
  owners.set(pipeline, owner);
  owners.set(proxy, owner);
  return proxy;
}
