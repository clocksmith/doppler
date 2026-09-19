import {
  runWithShaderSourceScope,
  streamWithShaderSourceScope,
} from '../../gpu/kernels/shader-source-scope.js';
import { applyPipelineContexts } from './context.js';
import { isDeviceLost } from '../../gpu/device-state.js';

function enterCompatibilityContext(pipeline, operation) {
  const lost = isDeviceLost(pipeline.gpuContext?.device);
  if (lost && operation !== 'unload' && operation !== 'cleanup') {
    throw new Error('Pipeline device is lost; reload this model on a live device.');
  }
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
  const methods = new Map();
  return new Proxy(pipeline, {
    get(target, key) {
      const value = Reflect.get(target, key, target);
      if (typeof value !== 'function') return value;
      if (methods.get(key)?.original === value) return methods.get(key).bound;
      // Text pipelines retain synchronous forwarding methods for their
      // generation controller. Preserve the controller's async/stream shape.
      const kind = value.constructor.name === 'Function' && typeof target.generator?.[key] === 'function'
        ? target.generator[key].constructor.name : value.constructor.name;
      let bound;
      if (kind === 'AsyncGeneratorFunction') {
        bound = (...args) => streamWithShaderSourceScope(scope, () => stream(target, key, () => value.apply(target, args)));
      } else if (kind === 'AsyncFunction') {
        bound = (...args) => runWithShaderSourceScope(scope, () => invoke(target, key, () => value.apply(target, args)));
      } else {
        bound = value.bind(target);
      }
      methods.set(key, { original: value, bound });
      return bound;
    },
  });
}
