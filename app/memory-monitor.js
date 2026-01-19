

import { getBufferPool } from '../src/memory/buffer-pool.js';

export class MemoryMonitor {
  #elements;

  #pollInterval = null;

  #pollIntervalMs = 2000;

  #estimatedSystemMemoryBytes = null;

  #gpuBufferLimitBytes = null;

  #isUnifiedMemory = false;

  #getPipelineMemoryStats = null;

  constructor(elements) {
    this.#elements = elements;
  }

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

  start() {
    this.update();
    this.#pollInterval = setInterval(() => this.update(), this.#pollIntervalMs);
  }

  stop() {
    if (this.#pollInterval) {
      clearInterval(this.#pollInterval);
      this.#pollInterval = null;
    }
  }

  update() {
    if (!this.#elements.heapBar) return;

    const stats = this.#gatherStats();
    this.#updateHeapUI(stats);
    this.#updateGPUUI(stats);
    this.#updateKVUI(stats);
    this.#updateOPFSUI();
    this.#updateTotalUI(stats);
  }

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

  #updateGPUUI(stats) {
    if (!this.#elements.gpuBar || !this.#elements.gpuValue) return;

    const minGpuLimit = 4 * 1024 * 1024 * 1024;
    const gpuLimit = this.#gpuBufferLimitBytes || Math.max(stats.peakGpuBytes, stats.usedGpuTotal, minGpuLimit);
    const gpuPercent = Math.min(100, (stats.usedGpuTotal / gpuLimit) * 100);

    this.#elements.gpuBar.style.width = `${gpuPercent}%`;
    this.#elements.gpuValue.textContent = this.formatBytes(stats.usedGpuTotal);
  }

  #updateKVUI(stats) {
    if (!this.#elements.kvBar || !this.#elements.kvValue) return;

    const minGpuLimit = 4 * 1024 * 1024 * 1024;
    const gpuLimit = this.#gpuBufferLimitBytes || Math.max(stats.peakGpuBytes, stats.usedGpuTotal, minGpuLimit);
    const kvPercent = Math.min(100, (stats.usedKv / gpuLimit) * 100);

    this.#elements.kvBar.style.width = `${kvPercent}%`;
    this.#elements.kvValue.textContent = stats.usedKv > 0 ? this.formatBytes(stats.usedKv) : '--';
  }

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

  #updateSwapIndicator(totalUsed) {
    if (!this.#elements.swapIndicator) return;

    const deviceMemoryGB = navigator.deviceMemory;
    if (deviceMemoryGB) {
      const physicalRamBytes = deviceMemoryGB * 1024 * 1024 * 1024;
      this.#elements.swapIndicator.hidden = totalUsed <= physicalRamBytes * 0.9;
    }
  }

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

  #getEstimatedSystemMemory() {
    return this.#estimatedSystemMemoryBytes;
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
}



