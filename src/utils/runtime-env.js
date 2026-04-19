export function isNodeRuntime() {
  return typeof process !== 'undefined'
    && process != null
    && typeof process === 'object'
    && process.versions != null
    && typeof process.versions.node === 'string';
}
