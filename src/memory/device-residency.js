// Buffer residency facts: which GPU device owns a buffer and which devices are
// lost. These are dependency-free accounting facts shared by the memory pools
// and the GPU device owner. The holder lives on globalThis so separately
// loaded module instances agree, as the device state holder does.
const RESIDENCY_KEY = '__dopplerDeviceResidency';
const observedDevices = new WeakSet();

/** @returns {import('./device-residency.js').DeviceResidency} */
function getResidency() {
  const scope = /** @type {{ [RESIDENCY_KEY]?: import('./device-residency.js').DeviceResidency }} */ (globalThis);
  const existing = scope[RESIDENCY_KEY];
  if (existing && typeof existing === 'object') return existing;
  const created = Object.freeze({ bufferOwners: new WeakMap(), lostDevices: new WeakSet() });
  Object.defineProperty(globalThis, RESIDENCY_KEY, {
    value: created,
    writable: false,
    enumerable: false,
    configurable: false,
  });
  return created;
}

/** @type {import('./device-residency.js').markDeviceLost} */
export function markDeviceLost(device) {
  if (device) getResidency().lostDevices.add(device);
}

/** @type {import('./device-residency.js').observeDeviceLoss} */
export function observeDeviceLoss(device) {
  if (!device || observedDevices.has(device)) return;
  observedDevices.add(device);
  const markLost = () => markDeviceLost(device);
  device.lost?.then(markLost, markLost);
}

/** @type {import('./device-residency.js').registerBufferDevice} */
export function registerBufferDevice(buffer, device) {
  if (!buffer || !device) throw new Error('Buffer ownership requires a buffer and device.');
  const owners = getResidency().bufferOwners;
  const previous = owners.get(buffer);
  if (previous && previous !== device) throw new Error('Cannot transfer a GPU buffer to another device.');
  observeDeviceLoss(device);
  owners.set(buffer, device);
}

// Records the creating device for a buffer returned by device.createBuffer().
/** @type {import('./device-residency.js').recordCreatedBuffer} */
export function recordCreatedBuffer(buffer, device) {
  getResidency().bufferOwners.set(buffer, device);
}

/** @type {import('./device-residency.js').getBufferDevice} */
export function getBufferDevice(buffer) {
  return getResidency().bufferOwners.get(buffer) ?? null;
}

/** @type {import('./device-residency.js').isDeviceLost} */
export function isDeviceLost(device) {
  return device != null && getResidency().lostDevices.has(device);
}
