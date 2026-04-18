import {
  getDevice,
  getKernelCapabilities,
  getPlatformConfig,
  setDevice,
} from '../../gpu/device.js';
import { applyDebugConfig, setGPUDevice } from '../../debug/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../config/runtime.js';
import { kernelTrace } from './text/kernel-trace.js';
import {
  getLogLevel,
  getTrace,
  isSilentMode,
  setLogLevel,
  setSilentMode,
  setTrace,
} from '../../debug/config.js';
import {
  gpuDevice as debugGpuDevice,
  traceBreakOnAnomaly,
  traceLayerFilter,
  traceMaxDecodeSteps,
} from '../../debug/config.js';

const RESTORE_PIPELINE_CONTEXTS = Symbol('restorePipelineContexts');

function captureTargetField(target, key) {
  return {
    present: Object.prototype.hasOwnProperty.call(target, key),
    value: target[key],
  };
}

function restoreTargetField(target, key, snapshot) {
  if (snapshot.present) {
    target[key] = snapshot.value;
    return;
  }
  delete target[key];
}

function captureDebugState() {
  return {
    logLevel: getLogLevel(),
    traceCategories: getTrace(),
    traceLayers: [...traceLayerFilter],
    traceMaxDecodeSteps,
    traceBreakOnAnomaly,
    silentMode: isSilentMode(),
    gpuDevice: debugGpuDevice,
  };
}

function restoreDebugState(snapshot) {
  if (snapshot.silentMode !== isSilentMode()) {
    setSilentMode(snapshot.silentMode);
  }
  if (getLogLevel() !== snapshot.logLevel) {
    setLogLevel(snapshot.logLevel);
  }

  const traceCategories = getTrace();
  const traceChanged = traceCategories.length !== snapshot.traceCategories.length
    || traceCategories.some((category, idx) => category !== snapshot.traceCategories[idx])
    || traceLayerFilter.length !== snapshot.traceLayers.length
    || traceLayerFilter.some((layer, idx) => layer !== snapshot.traceLayers[idx])
    || traceMaxDecodeSteps !== snapshot.traceMaxDecodeSteps
    || traceBreakOnAnomaly !== snapshot.traceBreakOnAnomaly;

  if (traceChanged) {
    if (snapshot.traceCategories.length > 0) {
      setTrace(snapshot.traceCategories.join(','), {
        layers: snapshot.traceLayers.length > 0 ? snapshot.traceLayers : undefined,
        maxDecodeSteps: snapshot.traceMaxDecodeSteps > 0 ? snapshot.traceMaxDecodeSteps : undefined,
        breakOnAnomaly: snapshot.traceBreakOnAnomaly,
      });
    } else {
      setTrace(false);
    }
  }

  setGPUDevice(snapshot.gpuDevice ?? null);
}

export function restorePipelineContexts(target) {
  const restore = target?.[RESTORE_PIPELINE_CONTEXTS];
  if (typeof restore !== 'function') {
    return false;
  }
  delete target[RESTORE_PIPELINE_CONTEXTS];
  restore();
  return true;
}

export function applyPipelineContexts(target, contexts = {}, options = {}) {
  restorePipelineContexts(target);

  const previousRuntimeConfig = getRuntimeConfig();
  const previousDevice = getDevice();
  const previousPlatformConfig = getPlatformConfig();
  const previousAdapterInfo = previousDevice
    ? (getKernelCapabilities().adapterInfo ?? null)
    : null;
  const previousDebugState = captureDebugState();
  const targetSnapshot = {
    gpuContext: captureTargetField(target, 'gpuContext'),
    useGPU: captureTargetField(target, 'useGPU'),
    memoryContext: captureTargetField(target, 'memoryContext'),
    storageContext: captureTargetField(target, 'storageContext'),
    baseUrl: captureTargetField(target, 'baseUrl'),
    _onProgress: captureTargetField(target, '_onProgress'),
  };

  const runtimeConfig = contexts.runtimeConfig
    ? setRuntimeConfig(contexts.runtimeConfig)
    : getRuntimeConfig();
  const sharedDebug = runtimeConfig.shared?.debug;

  if (options.applySharedDebug !== false && sharedDebug) {
    applyDebugConfig(sharedDebug);
    // Bridge runtime.shared.debug.kernelTrace → kernel-trace singleton.
    // Schema field has existed since gemma3-debug-q4k; bridging here so any
    // profile (verbose-trace, experiments/*) that sets it actually enables
    // the per-step NaN/Inf anomaly detector without a bespoke CLI flag.
    const kernelTraceCfg = sharedDebug.kernelTrace;
    console.error('[KT-BRIDGE] sharedDebug keys:', Object.keys(sharedDebug || {}).join(','), 'kernelTraceCfg:', JSON.stringify(kernelTraceCfg));
    if (kernelTraceCfg && (kernelTraceCfg.enabled === true || kernelTraceCfg.breakOnAnomaly === true)) {
      kernelTrace.enable({
        layers: kernelTraceCfg.layers ?? [],
        breakOnAnomaly: kernelTraceCfg.breakOnAnomaly ?? true,
        explosionThreshold: kernelTraceCfg.explosionThreshold ?? 10,
        collapseThreshold: kernelTraceCfg.collapseThreshold ?? 1e-6,
        maxSteps: kernelTraceCfg.maxSteps ?? 100,
      });
    } else if (kernelTrace.enabled && kernelTraceCfg?.enabled === false) {
      kernelTrace.disable();
    }
  }

  if (contexts.gpu?.device) {
    const device = contexts.gpu.device;
    const isRebindingCurrentDevice = previousDevice != null && device === previousDevice;
    const nextAdapterInfo = contexts.gpu.adapterInfo
      ?? (isRebindingCurrentDevice ? previousAdapterInfo : undefined);
    const nextPlatformConfig = contexts.gpu.platformConfig
      ?? (isRebindingCurrentDevice ? previousPlatformConfig : undefined);
    setDevice(device, {
      ...(nextAdapterInfo ? { adapterInfo: nextAdapterInfo } : {}),
      ...(nextPlatformConfig !== undefined ? { platformConfig: nextPlatformConfig } : {}),
    });
    setGPUDevice(device);
    if (options.assignGpuContext) {
      target.gpuContext = { device };
    }
    if (options.assignUseGPU) {
      target.useGPU = true;
    }
  } else {
    const device = getDevice();
    if (device) setGPUDevice(device);
  }

  if (options.assignMemoryContext && contexts.memory) {
    target.memoryContext = contexts.memory;
  }

  const requestedStorageContext = contexts.storage ?? contexts.storageContext;
  if (options.assignStorageContext && requestedStorageContext) {
    target.storageContext = requestedStorageContext;
  }
  if (options.assignBaseUrl !== false && contexts.baseUrl) {
    target.baseUrl = contexts.baseUrl;
  }
  if (options.assignProgress !== false && contexts.onProgress) {
    target._onProgress = contexts.onProgress;
  }

  let restored = false;
  const restore = () => {
    if (restored) {
      return;
    }
    restored = true;
    delete target[RESTORE_PIPELINE_CONTEXTS];

    setRuntimeConfig(previousRuntimeConfig);
    if (previousDevice) {
      setDevice(previousDevice, {
        platformConfig: previousPlatformConfig,
        adapterInfo: previousAdapterInfo,
      });
    } else {
      setDevice(null);
    }
    restoreDebugState(previousDebugState);
    restoreTargetField(target, 'gpuContext', targetSnapshot.gpuContext);
    restoreTargetField(target, 'useGPU', targetSnapshot.useGPU);
    restoreTargetField(target, 'memoryContext', targetSnapshot.memoryContext);
    restoreTargetField(target, 'storageContext', targetSnapshot.storageContext);
    restoreTargetField(target, 'baseUrl', targetSnapshot.baseUrl);
    restoreTargetField(target, '_onProgress', targetSnapshot._onProgress);
  };

  Object.defineProperty(target, RESTORE_PIPELINE_CONTEXTS, {
    value: restore,
    configurable: true,
    enumerable: false,
    writable: false,
  });

  return { runtimeConfig, sharedDebug, restore };
}
