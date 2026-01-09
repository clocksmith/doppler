/**
 * Hot-Swap Runtime State
 *
 * Manages the active hot-swap manifest for the current session.
 *
 * @module hotswap/runtime
 */

let activeManifest = null;

export function getHotSwapManifest() {
  return activeManifest;
}

export function setHotSwapManifest(manifest) {
  activeManifest = manifest;
}
