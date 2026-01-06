/**
 * Hot-Swap Runtime Registry
 *
 * Stores the active hot-swap manifest for the current session.
 *
 * @module hotswap/runtime
 */

import type { HotSwapManifest } from './manifest.js';

let activeManifest: HotSwapManifest | null = null;

export function getHotSwapManifest(): HotSwapManifest | null {
  return activeManifest;
}

export function setHotSwapManifest(manifest: HotSwapManifest | null): void {
  activeManifest = manifest;
}
