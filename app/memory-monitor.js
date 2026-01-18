

import { getBufferPool } from '../src/memory/buffer-pool.js';

/**
 * Monitors and reports memory usage (JS heap, GPU buffers, KV cache, OPFS).
 */
export class MemoryMonitor {
  /** @type {MemoryElements} */
  #elements;

  /** @type {number|null} */
  #pollInterval = null;

  /** @type {number} */
  #pollIntervalMs = 2000;

  /** @type {number|null} */
  #estimatedSystemMemoryBytes = null;

  /** @type {number|null} */
  #gpuBufferLimitBytes = null;

  /** @type {boolean} */
  #isUnifiedMemory = false;

  /** @type {(() => object|null)|null} */
  #getPipelineMemoryStats = null;

  /**
   * @param {MemoryElements} elements - DOM elements for memory display
   */
  constructor(elements) {
    this.#elements = elements;
  }

  /**
   * Configure the monitor.
   * @param {MemoryMonitorConfig} config
   */
  configure(config) {
    if (config.estimatedSystemMemoryBytes !== undefined) {
      this.#estimatedSystemMemoryBytes = config.estimatedSystemMemoryBytes;
    }
    if (config.gpuBufferLimitBytes !== undefined) {
      this.#gpuBufferLimitBytes = config.gpuBufferLimitBytes;
    }
    if (config.isUnifiedMemory !== undefined) {
      this.#isUnifiedMemory = config.isUnifiedMemory;
    }
    if (config.getPipelineMemoryStats !== undefined) {
      this.#getPipelineMemoryStats = config.getPipelineMemoryStats;
    }
    if (config.pollIntervalMs !== undefined) {
      this.#pollIntervalMs = config.pollIntervalMs;
    }
  }

  /**
   * Start memory polling.
   */
  start() {
    this.update();
    this.#pollInterval = setInterval(() => this.update(), this.#pollIntervalMs);
  }

  /**
   * Stop memory polling.
   */
  stop() {
    if (this.#pollInterval) {
      clearInterval(this.#pollInterval);
      this.#pollInterval = null;
    }
  }

  /**
   * Update all memory stats.
   */
  update() {
    if (!this.#elements.heapBar) return;

    const stats = this.#gatherStats();
    this.#updateHeapUI(stats);
    this.#updateGPUUI(stats);
    this.#updateKVUI(stats);
    this.#updateOPFSUI();
    this.#updateTotalUI(stats);
  }

  /**
   * Gather current memory statistics.
   * @returns {MemoryStats}
   */
  #gatherStats() {
    let usedHeap = 0;
    let totalHeapLimit = 0;
    let usedGpuPool = 0;
    let usedKv = 0;
    let peakGpuBytes = 0;

    // JS Heap (Chrome only)
    const memory = performance.memory;
    if (memory) {
      usedHeap = memory.usedJSHeapSize || 0;
      totalHeapLimit = memory.jsHeapSizeLimit || memory.totalJSHeapSize || 1;
    }

    // GPU buffer usage
    try {
      const bufferPool = getBufferPool();
      const poolStats = bufferPool.getStats();
      usedGpuPool = poolStats.currentBytesAllocated || 0;
      peakGpuBytes = poolStats.peakBytesAllocated || 0;
    } catch {
      // Buffer pool not initialized
    }

    // KV cache usage
    if (this.#getPipelineMemoryStats) {
      const memStats = this.#getPipelineMemoryStats();
      if (memStats?.kvCache?.allocated) {
        usedKv = memStats.kvCache.allocated;
      }
    }

    return {
      usedHeap,
      totalHeapLimit,
      usedGpuPool,
      usedKv,
      peakGpuBytes,
      usedGpuTotal: usedGpuPool + usedKv,
    };
  }

  /**
   * Update heap UI elements.
   * @param {MemoryStats} stats
   */
  #updateHeapUI(stats) {
    const memory = performance.memory;
    if (memory && this.#elements.heapBar && this.#elements.heapValue) {
      const heapPercent = Math.min(100, (stats.usedHeap / stats.totalHeapLimit) * 100);
      this.#elements.heapBar.style.width = `${heapPercent}%`;
      this.#elements.heapValue.textContent = this.formatBytes(stats.usedHeap);
    } else if (this.#elements.heapValue) {
      this.#elements.heapValue.textContent = 'N/A';
    }
  }

  /**
   * Update GPU UI elements.
   * @param {MemoryStats} stats
   */
  #updateGPUUI(stats) {
    if (!this.#elements.gpuBar || !this.#elements.gpuValue) return;

    const minGpuLimit = 4 * 1024 * 1024 * 1024;
    const gpuLimit = this.#gpuBufferLimitBytes || Math.max(stats.peakGpuBytes, stats.usedGpuTotal, minGpuLimit);
    const gpuPercent = Math.min(100, (stats.usedGpuTotal / gpuLimit) * 100);

    this.#elements.gpuBar.style.width = `${gpuPercent}%`;
    this.#elements.gpuValue.textContent = this.formatBytes(stats.usedGpuTotal);
  }

  /**
   * Update KV cache UI elements.
   * @param {MemoryStats} stats
   */
  #updateKVUI(stats) {
    if (!this.#elements.kvBar || !this.#elements.kvValue) return;

    const minGpuLimit = 4 * 1024 * 1024 * 1024;
    const gpuLimit = this.#gpuBufferLimitBytes || Math.max(stats.peakGpuBytes, stats.usedGpuTotal, minGpuLimit);
    const kvPercent = Math.min(100, (stats.usedKv / gpuLimit) * 100);

    this.#elements.kvBar.style.width = `${kvPercent}%`;
    this.#elements.kvValue.textContent = stats.usedKv > 0 ? this.formatBytes(stats.usedKv) : '--';
  }

  /**
   * Update OPFS UI elements (async).
   */
  #updateOPFSUI() {
    if (!this.#elements.opfsBar || !this.#elements.opfsValue) return;

    navigator.storage.estimate().then((estimate) => {
      const opfsUsed = estimate.usage || 0;
      const opfsQuota = estimate.quota || 1;
      const opfsPercent = Math.min(100, (opfsUsed / opfsQuota) * 100);

      this.#elements.opfsBar.style.width = `${opfsPercent}%`;
      this.#elements.opfsValue.textContent = this.formatBytes(opfsUsed);
    }).catch(() => {
      this.#elements.opfsValue.textContent = '--';
    });
  }

  /**
   * Update total/stacked memory UI.
   * @param {MemoryStats} stats
   */
  #updateTotalUI(stats) {
    if (!this.#elements.heapStackedBar || !this.#elements.gpuStackedBar) return;

    const totalUsed = stats.usedHeap + stats.usedGpuTotal;
    const combinedLimit = Math.max(stats.totalHeapLimit, 8 * 1024 * 1024 * 1024);
    const heapStackedPercent = Math.min(50, (stats.usedHeap / combinedLimit) * 100);
    const gpuStackedPercent = Math.min(50, (stats.usedGpuTotal / combinedLimit) * 100);

    this.#elements.heapStackedBar.style.width = `${heapStackedPercent}%`;
    this.#elements.gpuStackedBar.style.width = `${gpuStackedPercent}%`;

    if (this.#elements.totalValue) {
      this.#elements.totalValue.textContent = this.formatBytes(totalUsed);
    }

    // Swap warning
    this.#updateSwapIndicator(totalUsed);

    // Headroom
    this.#updateHeadroomUI(totalUsed);
  }

  /**
   * Update swap indicator visibility.
   * @param {number} totalUsed
   */
  #updateSwapIndicator(totalUsed) {
    if (!this.#elements.swapIndicator) return;

    const deviceMemoryGB = navigator.deviceMemory;
    if (deviceMemoryGB) {
      const physicalRamBytes = deviceMemoryGB * 1024 * 1024 * 1024;
      this.#elements.swapIndicator.hidden = totalUsed <= physicalRamBytes * 0.9;
    }
  }

  /**
   * Update headroom UI.
   * @param {number} totalUsed
   */
  #updateHeadroomUI(totalUsed) {
    if (!this.#elements.headroomValue || !this.#elements.headroomBar) return;

    const estimatedTotal = this.#getEstimatedSystemMemory();
    if (!this.#isUnifiedMemory || !estimatedTotal) {
      this.#elements.headroomValue.textContent = '--';
      this.#elements.headroomBar.style.width = '0%';
      return;
    }

    const headroomBytes = Math.max(0, estimatedTotal - totalUsed);
    const headroomPercent = Math.min(100, (headroomBytes / estimatedTotal) * 100);
    this.#elements.headroomValue.textContent = this.formatBytes(headroomBytes);
    this.#elements.headroomBar.style.width = `${headroomPercent}%`;
  }

  /**
   * Get estimated system memory.
   * @returns {number|null}
   */
  #getEstimatedSystemMemory() {
    return this.#estimatedSystemMemoryBytes;
  }

  /**
   * Format bytes to human-readable string.
   * @param {number} bytes
   * @returns {string}
   */
  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
}

/**
 * @typedef {Object} MemoryElements
 * @property {HTMLElement|null} heapBar
 * @property {HTMLElement|null} heapValue
 * @property {HTMLElement|null} gpuBar
 * @property {HTMLElement|null} gpuValue
 * @property {HTMLElement|null} kvBar
 * @property {HTMLElement|null} kvValue
 * @property {HTMLElement|null} opfsBar
 * @property {HTMLElement|null} opfsValue
 * @property {HTMLElement|null} headroomBar
 * @property {HTMLElement|null} headroomValue
 * @property {HTMLElement|null} heapStackedBar
 * @property {HTMLElement|null} gpuStackedBar
 * @property {HTMLElement|null} totalValue
 * @property {HTMLElement|null} swapIndicator
 */

/**
 * @typedef {Object} MemoryMonitorConfig
 * @property {number} [estimatedSystemMemoryBytes]
 * @property {number} [gpuBufferLimitBytes]
 * @property {boolean} [isUnifiedMemory]
 * @property {() => object|null} [getPipelineMemoryStats]
 * @property {number} [pollIntervalMs]
 */

/**
 * @typedef {Object} MemoryStats
 * @property {number} usedHeap
 * @property {number} totalHeapLimit
 * @property {number} usedGpuPool
 * @property {number} usedKv
 * @property {number} peakGpuBytes
 * @property {number} usedGpuTotal
 */
