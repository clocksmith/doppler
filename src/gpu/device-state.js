const SHARED_DEVICE_STATE_KEY = '__dopplerGpuDeviceState';

export function getSharedDeviceState() {
  const existing = globalThis[SHARED_DEVICE_STATE_KEY];
  if (existing && typeof existing === 'object') {
    if (!(existing.bufferOwners instanceof WeakMap)) {
      existing.bufferOwners = new WeakMap();
    }
    if (!(existing.lostDevices instanceof WeakSet)) {
      existing.lostDevices = new WeakSet();
    }
    if (!('deviceInitPromise' in existing)) {
      existing.deviceInitPromise = null;
    }
    return existing;
  }
  const created = {
    gpuDevice: null,
    kernelCapabilities: null,
    resolvedPlatformConfig: null,
    lastDeviceLossInfo: null,
    platformInitialized: false,
    deviceEpoch: 0,
    bufferOwners: new WeakMap(),
    lostDevices: new WeakSet(),
    deviceInitPromise: null,
  };
  Object.defineProperty(globalThis, SHARED_DEVICE_STATE_KEY, {
    value: created,
    writable: false,
    enumerable: false,
    configurable: false,
  });
  return created;
}

export function getSharedDeviceEpoch() {
  const epoch = getSharedDeviceState().deviceEpoch;
  return Number.isInteger(epoch) ? epoch : 0;
}

export function isDeviceLost(device) {
  return device != null && getSharedDeviceState().lostDevices.has(device);
}
