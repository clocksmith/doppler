import {
  runWithShaderSourceScope,
  streamWithShaderSourceScope,
} from '../../gpu/kernels/shader-source-scope.js';

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
        bound = (...args) => streamWithShaderSourceScope(scope, () => value.apply(target, args));
      } else if (kind === 'AsyncFunction') {
        bound = (...args) => runWithShaderSourceScope(scope, () => value.apply(target, args));
      } else {
        bound = value.bind(target);
      }
      methods.set(key, { original: value, bound });
      return bound;
    },
  });
}
