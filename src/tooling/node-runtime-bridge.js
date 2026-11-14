import { installNodeFileFetchShim } from './node-file-fetch.js';
import { loadRuntimeConfigFromRef } from './browser-harness/runtime-config.js';
import {
  getActiveKernelPath,
  getActiveKernelPathPolicy,
  getActiveKernelPathSource,
  setActiveKernelPath,
} from '../config/kernel-path-loader.js';

/** @type {Promise<import('./node-runtime-bridge.js').NodeRuntimeModules> | null} */
let runtimeModulesPromise = null;

/** @type {typeof import('./node-runtime-bridge.js').loadRuntimeModules} */
export function loadRuntimeModules() {
  if (runtimeModulesPromise) {
    return runtimeModulesPromise;
  }

  installNodeFileFetchShim();
  runtimeModulesPromise = Promise.all([
    import('./browser-harness.js'),
    import('../config/runtime.js'),
  ]).then(([harness, runtime]) => ({ harness, runtime }));

  return runtimeModulesPromise;
}

/** @type {typeof import('./node-runtime-bridge.js').createRuntimeBridge} */
export function createRuntimeBridge(modules) {
  return {
    loadRuntimeConfigFromRef,
    loadRuntimeProfile: modules.harness.loadRuntimeProfile,
    loadRuntimeConfigFromUrl: modules.harness.loadRuntimeConfigFromUrl,
    getRuntimeConfig: modules.runtime.getRuntimeConfig,
    setRuntimeConfig: modules.runtime.setRuntimeConfig,
    resetRuntimeConfig: modules.runtime.resetRuntimeConfig,
    getActiveKernelPath,
    getActiveKernelPathPolicy,
    getActiveKernelPathSource,
    setActiveKernelPath,
  };
}

