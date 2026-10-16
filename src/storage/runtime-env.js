export function isNodeRuntime() {
  return typeof process !== 'undefined'
    && process != null
    && typeof process === 'object'
    && process.versions != null
    && typeof process.versions.node === 'string';
}

// Some embedded hosts expose a Node version without satisfying the stricter
// runtime contract above. Preserve that presence check for those integrations.
export function hasNodeVersion() {
  return typeof process !== 'undefined' && !!process.versions?.node;
}
