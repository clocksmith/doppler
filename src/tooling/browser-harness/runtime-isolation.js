import { cloneJsonValue } from '../../formats/clone-json.js';
import {
  getActiveKernelPath,
  getActiveKernelPathPolicy,
  getActiveKernelPathSource,
  setActiveKernelPath,
} from '../../config/kernel-path-loader.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../config/runtime.js';

export function cloneRuntimeConfig(runtimeConfig) {
  return runtimeConfig ? cloneJsonValue(runtimeConfig) : null;
}

export function snapshotRuntimeState() {
  return {
    runtimeConfig: cloneRuntimeConfig(getRuntimeConfig()),
    activeKernelPath: getActiveKernelPath(),
    activeKernelPathSource: getActiveKernelPathSource(),
    activeKernelPathPolicy: getActiveKernelPathPolicy(),
  };
}

export function restoreRuntimeState(snapshot) {
  if (!snapshot) {
    return;
  }
  setRuntimeConfig(snapshot.runtimeConfig);
  setActiveKernelPath(
    snapshot.activeKernelPath,
    snapshot.activeKernelPathSource || 'none',
    snapshot.activeKernelPathPolicy ?? null
  );
}

export async function runWithRuntimeIsolationForSuite(run) {
  const snapshot = snapshotRuntimeState();
  try {
    return await run();
  } finally {
    restoreRuntimeState(snapshot);
  }
}
