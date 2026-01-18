

import { log } from '../src/debug/index.js';

/**
 * Detects and reports browser/GPU capabilities for the demo app.
 */
export class CapabilitiesDetector {
  /** @type {CapabilitiesState} */
  #state = {
    webgpu: false,
    f16: false,
    subgroups: false,
    memory64: false,
  };

  /** @type {GPUAdapterInfo|null} */
  #adapterInfo = null;

  /** @type {GPUAdapter|null} */
  #adapter = null;

  /**
   * Detect all capabilities.
   * @returns {Promise<CapabilitiesState>}
   */
  async detect() {
    log.debug('Capabilities', 'Detecting capabilities...');

    await this.#detectWebGPU();
    await this.#detectMemory64();

    return this.#state;
  }

  /**
   * Detect WebGPU and related features.
   */
  async #detectWebGPU() {
    if (!navigator.gpu) {
      return;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        return;
      }

      this.#adapter = adapter;
      this.#state.webgpu = true;
      this.#state.f16 = adapter.features.has('shader-f16');
      this.#state.subgroups = adapter.features.has('subgroups');

      // Get adapter info
      this.#adapterInfo = adapter.info || (await adapter.requestAdapterInfo?.()) || {};
      log.info('GPU', `${this.#adapterInfo.vendor || 'unknown'} ${this.#adapterInfo.architecture || this.#adapterInfo.device || 'unknown'}`);
    } catch (e) {
      log.warn('Capabilities', 'WebGPU init failed:', e);
    }
  }

  /**
   * Detect WebAssembly Memory64 support.
   */
  async #detectMemory64() {
    try {
      const memory64Test = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x05, 0x04, 0x01, 0x04, 0x01, 0x00,
      ]);
      await WebAssembly.compile(memory64Test);
      this.#state.memory64 = true;
    } catch {
      this.#state.memory64 = false;
    }
  }

  /**
   * Get current capabilities state.
   * @returns {CapabilitiesState}
   */
  getState() {
    return { ...this.#state };
  }

  /**
   * Get the GPU adapter (if available).
   * @returns {GPUAdapter|null}
   */
  getAdapter() {
    return this.#adapter;
  }

  /**
   * Get adapter info (if available).
   * @returns {GPUAdapterInfo|null}
   */
  getAdapterInfo() {
    return this.#adapterInfo;
  }

  /**
   * Resolve a human-readable GPU name from adapter info.
   * @param {GPUAdapterInfo} info
   * @returns {string}
   */
  resolveGPUName(info) {
    const vendor = (info.vendor || '').toLowerCase();
    const device = (info.device || '').toLowerCase();
    const arch = (info.architecture || '').toLowerCase();

    // Try parsing architecture string (works well on Apple Silicon)
    if (arch) {
      const appleMatch = arch.match(/apple[- ]?(m\d+)(?:[- ]?(pro|max|ultra))?/i);
      if (appleMatch) {
        const chip = appleMatch[1].toUpperCase();
        const variant = appleMatch[2]
          ? ` ${appleMatch[2].charAt(0).toUpperCase() + appleMatch[2].slice(1)}`
          : '';
        return `Apple ${chip}${variant}`;
      }
      if (arch.length > 3 && !arch.startsWith('0x')) {
        return arch.split('-').map(s => s.charAt(0).toUpperCase() + s.slice(1)).join(' ');
      }
    }

    // Try description field
    if (info.description && info.description.length > 3) {
      return info.description;
    }

    // Last resort: vendor + device
    if (vendor && device) {
      log.info('GPU', `Unknown device: vendor=${vendor}, device=${device}, arch=${arch}`);
      const vendorName = vendor.charAt(0).toUpperCase() + vendor.slice(1);
      return `${vendorName} GPU`;
    }

    return 'Unknown GPU';
  }

  /**
   * Check if the system uses unified memory architecture (Apple Silicon, etc).
   * @param {GPUAdapterInfo} info
   * @returns {boolean}
   */
  isUnifiedMemoryArchitecture(info) {
    const arch = info.architecture?.toLowerCase() || '';
    const vendor = info.vendor?.toLowerCase() || '';
    const desc = info.description?.toLowerCase() || '';

    // Apple Silicon uses unified memory
    if (vendor.includes('apple') || arch.includes('apple') || desc.includes('apple')) {
      return true;
    }

    // Metal GPU on macOS is unified memory
    if (desc.includes('metal')) {
      return true;
    }

    // Check platform for macOS with ARM (Apple Silicon)
    const ua = navigator.userAgent.toLowerCase();
    if (ua.includes('mac') && (ua.includes('arm') || navigator.platform === 'MacIntel')) {
      if (desc.includes('metal') || vendor.includes('apple')) {
        return true;
      }
    }

    return false;
  }

  /**
   * Get GPU limits from the adapter.
   * @returns {{maxBufferSize: number, maxStorageSize: number}|null}
   */
  getGPULimits() {
    if (!this.#adapter) {
      return null;
    }

    const limits = this.#adapter.limits || {};
    return {
      maxBufferSize: limits.maxBufferSize || 0,
      maxStorageSize: limits.maxStorageBufferBindingSize || 0,
    };
  }

  /**
   * Check if timestamp queries are supported.
   * @returns {boolean}
   */
  hasTimestampQuery() {
    return this.#adapter?.features.has('timestamp-query') ?? false;
  }
}

/**
 * @typedef {Object} CapabilitiesState
 * @property {boolean} webgpu
 * @property {boolean} f16
 * @property {boolean} subgroups
 * @property {boolean} memory64
 */
