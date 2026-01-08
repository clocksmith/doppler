/**
 * app.js - DOPPLER Application Controller
 *
 * Main application that wires together all components and the DOPPLER inference pipeline.
 *
 * @module app/app
 */

import { ModelSelector } from './model-selector.js';
import { ChatUI } from './chat-ui.js';
import { ProgressUI } from './progress-ui.js';
import { QuickStartUI } from './quickstart-ui.js';
import { log } from '../src/debug/index.js';

// Quick-start downloader
import {
  downloadQuickStartModel,
  QUICKSTART_MODELS,
} from '../src/storage/quickstart-downloader.js';

// Browser model converter
import {
  convertModel,
  pickModelFiles,
  isConversionSupported,
  ConvertStage,
} from '../src/browser/browser-converter.js';

// DOPPLER pipeline imports
import { createPipeline } from '../src/inference/pipeline.js';
import { downloadModel } from '../src/storage/downloader.js';
import {
  listModels,
  openModelDirectory,
  loadManifestFromOPFS,
  deleteModel as deleteModelFromOPFS,
} from '../src/storage/shard-manager.js';
import { parseManifest } from '../src/storage/rdrr-format.js';
import { getMemoryCapabilities } from '../src/memory/capability.js';
import { getHeapManager } from '../src/memory/heap-manager.js';
import { getBufferPool } from '../src/gpu/buffer-pool.js';
import { initDevice, getKernelCapabilities, getDevice } from '../src/gpu/device.js';

// ============================================================================
// Constants
// ============================================================================

/**
 * Remote models available for download
 * Currently empty - only local models (discovered via /api/models) are shown
 * @type {import('./app.js').RemoteModel[]}
 */
const REMOTE_MODELS = [];

/**
 * Dynamic model registry populated at runtime
 * @type {import('./app.js').RegisteredModel[]}
 */
let MODEL_REGISTRY = [];

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Discover local models via server API
 * @returns {Promise<import('./app.js').RemoteModel[]>}
 */
async function discoverLocalModels() {
  const baseUrl = window.location.origin;

  try {
    const response = await fetch(`${baseUrl}/api/models`);
    if (!response.ok) return [];

    /** @type {import('./app.js').ServerModel[]} */
    const models = await response.json();
    return models.map((m) => {
      // Create friendly name from folder name
      let modelName = m.name
        .replace(/-rdrr$/, '')
        .replace(/-q4$/, '')
        .split('-')
        .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
        .join(' ');

      // Infer param count from layers/hidden
      const inferredParams = m.size || (m.numLayers ? `${m.numLayers}L` : 'Unknown');

      return {
        id: m.name,
        name: modelName,
        size: inferredParams,
        quantization: m.quantization || 'Unknown',
        downloadSize: m.downloadSize || 0,
        url: `${baseUrl}/${m.path}`,
        source: 'local',
        downloaded: true,
        architecture: m.architecture || 'Unknown',
      };
    });
  } catch (e) {
    log.debug('Discovery', 'Failed to fetch models from API:', e);
    return [];
  }
}

// ============================================================================
// Main Demo Application
// ============================================================================

/**
 * Main Demo Application
 */
export class DopplerDemo {
  /** @type {ModelSelector | null} */
  #modelSelector = null;
  /** @type {ChatUI | null} */
  #chatUI = null;
  /** @type {ProgressUI | null} */
  #progressUI = null;
  /** @type {QuickStartUI | null} */
  #quickStartUI = null;

  // Pipeline state
  /** @type {import('../src/inference/pipeline.js').Pipeline | null} */
  #pipeline = null;
  /** @type {import('./app.js').RegisteredModel | null} */
  #currentModel = null;
  /** @type {boolean} */
  #isGenerating = false;
  /** @type {AbortController | null} */
  #abortController = null;

  // Capabilities
  /** @type {import('./app.js').Capabilities} */
  #capabilities = {
    webgpu: false,
    f16: false,
    subgroups: false,
    memory64: false,
  };

  // DOM references
  /** @type {HTMLElement | null} */
  #statusDot = null;
  /** @type {HTMLElement | null} */
  #statusText = null;
  /** @type {HTMLElement | null} */
  #capabilitiesList = null;
  /** @type {import('./app.js').StatsElements} */
  #statsElements = {
    tps: null,
    memory: null,
    gpu: null,
    kv: null,
  };

  // GPU info elements
  /** @type {import('./app.js').GPUElements} */
  #gpuElements = {
    device: null,
    vram: null,
    vramLabel: null,
    ram: null,
    ramRow: null,
    features: null,
    unifiedNote: null,
  };

  // Memory bar elements
  /** @type {import('./app.js').MemoryElements} */
  #memoryElements = {
    heapBar: null,
    heapValue: null,
    gpuBar: null,
    gpuValue: null,
    kvBar: null,
    kvValue: null,
    opfsBar: null,
    opfsValue: null,
    headroomBar: null,
    headroomValue: null,
    heapStackedBar: null,
    gpuStackedBar: null,
    totalValue: null,
  };
  /** @type {import('../src/memory/capability.js').MemoryCapabilities | null} */
  #memoryCapabilities = null;
  /** @type {number | null} */
  #estimatedSystemMemoryBytes = null;
  /** @type {number | null} */
  #gpuBufferLimitBytes = null;
  /** @type {boolean} */
  #isUnifiedMemory = false;

  // Memory control UI
  /** @type {HTMLButtonElement | null} */
  #unloadModelBtn = null;
  /** @type {HTMLButtonElement | null} */
  #clearMemoryBtn = null;
  /** @type {HTMLElement | null} */
  #swapIndicator = null;
  /** @type {import('../src/config/schema/index.js').KernelPathRef | null} */
  #runtimeKernelPath = null;

  // Sampling controls
  /** @type {HTMLInputElement | null} */
  #temperatureInput = null;
  /** @type {HTMLInputElement | null} */
  #topPInput = null;
  /** @type {HTMLInputElement | null} */
  #topKInput = null;

  // Converter UI
  /** @type {HTMLButtonElement | null} */
  #convertBtn = null;
  /** @type {HTMLElement | null} */
  #convertStatus = null;
  /** @type {HTMLElement | null} */
  #convertProgress = null;
  /** @type {HTMLElement | null} */
  #convertMessage = null;
  /** @type {boolean} */
  #isConverting = false;

  // Memory polling interval
  /** @type {ReturnType<typeof setInterval> | null} */
  #memoryPollInterval = null;

  /**
   * Initialize the application
   * @returns {Promise<void>}
   */
  async init() {
    log.info('App', 'Initializing...');

    // Get DOM references
    this.#statusDot = document.querySelector('.status-dot');
    this.#statusText = document.querySelector('.status-text');
    this.#capabilitiesList = document.querySelector('#capabilities-list');
    this.#statsElements = {
      tps: document.querySelector('#stat-tps'),
      memory: document.querySelector('#stat-memory'),
      gpu: document.querySelector('#stat-gpu'),
      kv: document.querySelector('#stat-kv'),
    };

    // GPU info elements
    this.#gpuElements = {
      device: document.querySelector('#gpu-device'),
      vram: document.querySelector('#gpu-vram'),
      vramLabel: document.querySelector('#gpu-vram-label'),
      ram: document.querySelector('#gpu-ram'),
      ramRow: document.querySelector('#gpu-ram-row'),
      features: document.querySelector('#gpu-features'),
      unifiedNote: document.querySelector('#gpu-unified-note'),
    };

    // Memory bar elements
    this.#memoryElements = {
      heapBar: document.querySelector('#memory-bar-heap'),
      heapValue: document.querySelector('#memory-heap'),
      gpuBar: document.querySelector('#memory-bar-gpu'),
      gpuValue: document.querySelector('#memory-gpu'),
      kvBar: document.querySelector('#memory-bar-kv'),
      kvValue: document.querySelector('#memory-kv'),
      opfsBar: document.querySelector('#memory-bar-opfs'),
      opfsValue: document.querySelector('#memory-opfs'),
      headroomBar: document.querySelector('#memory-bar-headroom'),
      headroomValue: document.querySelector('#memory-headroom'),
      // Stacked total bar
      heapStackedBar: document.querySelector('#memory-bar-heap-stacked'),
      gpuStackedBar: document.querySelector('#memory-bar-gpu-stacked'),
      totalValue: document.querySelector('#memory-total'),
    };

    this.#readRuntimeOverridesFromURL();

    // Memory control elements
    this.#unloadModelBtn = document.querySelector('#unload-model-btn');
    this.#clearMemoryBtn = document.querySelector('#clear-memory-btn');
    this.#swapIndicator = document.querySelector('#swap-indicator');

    this.#temperatureInput = document.querySelector('#temperature-input');
    this.#topPInput = document.querySelector('#top-p-input');
    this.#topKInput = document.querySelector('#top-k-input');

    // Converter elements
    this.#convertBtn = document.querySelector('#convert-btn');
    this.#convertStatus = document.querySelector('#convert-status');
    this.#convertProgress = document.querySelector('#convert-progress');
    this.#convertMessage = document.querySelector('#convert-message');

    // Initialize UI components
    this.#initComponents();

    // Check WebGPU support
    await this.#detectCapabilities();

    // Load cached models list
    await this.#loadCachedModels();

    // Set initial status
    if (this.#capabilities.webgpu) {
      this.#setStatus('ready', 'Ready');
      this.#chatUI?.setInputEnabled(false); // Disabled until model loaded
    } else {
      this.#setStatus('error', 'WebGPU not supported');
      this.#showError(
        'WebGPU is not available in this browser. Please use Chrome 113+, Edge 113+, or Firefox Nightly with WebGPU enabled.'
      );
    }

    log.info('App', 'Initialized');
  }

  /**
   * Initialize UI components
   */
  #initComponents() {
    const container = /** @type {HTMLElement} */ (document.querySelector('#app'));

    // Model Selector
    this.#modelSelector = new ModelSelector(container, {
      onSelect: (model, opts) => this.selectModel(/** @type {import('./app.js').RegisteredModel} */ (model), opts),
      onDownload: (model, opts) => this.downloadModel(/** @type {import('./app.js').RegisteredModel} */ (model), opts),
      onDelete: (model) => this.deleteModel(/** @type {import('./app.js').RegisteredModel} */ (model)),
      onQuickStart: (model) => {
        // Use the remote source ID (e.g., 'gemma-1b-instruct') for QUICKSTART_MODELS lookup
        const modelId = model.sources?.remote?.id || model.key;
        this.startQuickStart(modelId);
      },
    });

    // Chat UI
    this.#chatUI = new ChatUI(container, {
      onSend: (message) => this.chat(message),
      onStop: () => this.stopGeneration(),
      onClear: () => this.clearConversation(),
    });

    // Progress UI
    this.#progressUI = new ProgressUI(container);

    // Quick-Start UI
    this.#quickStartUI = new QuickStartUI(container, {
      onDownloadComplete: (modelId) => this.#onQuickStartComplete(modelId),
      onRunModel: (modelId) => this.#runQuickStartModel(modelId),
      onCancel: () => log.debug('QuickStart', 'Cancelled by user'),
    });

    // Sampling inputs (clamp and persist)
    /**
     * @param {HTMLInputElement} input
     * @param {number} min
     * @param {number} max
     */
    const clampNumber = (input, min, max) => {
      const n = parseFloat(input.value);
      if (!Number.isFinite(n)) return;
      input.value = Math.min(max, Math.max(min, n)).toString();
    };
    if (this.#temperatureInput) {
      this.#temperatureInput.addEventListener('change', () =>
        clampNumber(/** @type {HTMLInputElement} */ (this.#temperatureInput), 0.1, 2.0)
      );
    }
    if (this.#topPInput) {
      this.#topPInput.addEventListener('change', () => clampNumber(/** @type {HTMLInputElement} */ (this.#topPInput), 0, 1));
    }
    if (this.#topKInput) {
      this.#topKInput.addEventListener('change', () => clampNumber(/** @type {HTMLInputElement} */ (this.#topKInput), 0, 200));
    }

    // Convert button
    if (this.#convertBtn) {
      if (isConversionSupported()) {
        this.#convertBtn.addEventListener('click', () => this.#handleConvert());
      } else {
        this.#convertBtn.disabled = true;
        this.#convertBtn.title = 'Model conversion requires File System Access API (Chrome/Edge)';
      }
    }

    // Memory control buttons
    if (this.#unloadModelBtn) {
      this.#unloadModelBtn.addEventListener('click', () => this.#unloadCurrentModel());
    }
    if (this.#clearMemoryBtn) {
      this.#clearMemoryBtn.addEventListener('click', () => this.#clearAllMemory());
    }
  }

  #readRuntimeOverridesFromURL() {
    const params = new URLSearchParams(window.location.search);
    const kernelPathRaw = params.get('kernelPath');
    if (kernelPathRaw) {
      try {
        const trimmed = kernelPathRaw.trim();
        if (trimmed.startsWith('{')) {
          const parsed = JSON.parse(trimmed);
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            this.#runtimeKernelPath = /** @type {import('../src/config/schema/index.js').KernelPathRef} */ (parsed);
            log.debug('App', 'Runtime kernel path from URL:', parsed);
          } else {
            log.warn('App', 'kernelPath JSON must be an object; ignoring.');
          }
        } else {
          this.#runtimeKernelPath = trimmed;
          log.debug('App', 'Runtime kernel path from URL:', trimmed);
        }
      } catch (err) {
        log.warn('App', 'Failed to parse kernelPath from URL:', /** @type {Error} */ (err).message);
      }
    }
  }

  /**
   * Detect browser capabilities
   * @returns {Promise<void>}
   */
  async #detectCapabilities() {
    log.debug('App', 'Detecting capabilities...');

    // WebGPU
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          this.#capabilities.webgpu = true;

          // Check features
          this.#capabilities.f16 = adapter.features.has('shader-f16');
          this.#capabilities.subgroups = adapter.features.has('subgroups');

          // Get adapter info for logging
          /** @type {Partial<GPUAdapterInfo>} */
          const info = /** @type {GPUAdapter & { info?: GPUAdapterInfo; requestAdapterInfo?: () => Promise<GPUAdapterInfo> }} */ (adapter).info || (await /** @type {GPUAdapter & { requestAdapterInfo?: () => Promise<GPUAdapterInfo> }} */ (adapter).requestAdapterInfo?.()) || {};
          log.info('GPU', `${info.vendor || 'unknown'} ${info.architecture || info.device || 'unknown'}`);

          // Populate GPU info panel
          this.#populateGPUInfo(adapter, /** @type {GPUAdapterInfo} */ (info));
        }
      } catch (e) {
        log.warn('App', 'WebGPU init failed:', e);
      }
    }

    // Start memory stats polling
    this.#startMemoryPolling();

    // Memory64 (basic check)
    try {
      const memory64Test = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x05, 0x04, 0x01, 0x04, 0x01, 0x00,
      ]);
      await WebAssembly.compile(memory64Test);
      this.#capabilities.memory64 = true;
    } catch {
      this.#capabilities.memory64 = false;
    }

    // Update UI
    this.#updateCapabilitiesUI();
  }

  /**
   * Update capabilities list UI
   */
  #updateCapabilitiesUI() {
    if (!this.#capabilitiesList) return;
    const items = this.#capabilitiesList.querySelectorAll('li');
    items.forEach((item) => {
      const cap = /** @type {keyof import('./app.js').Capabilities} */ (/** @type {HTMLElement} */ (item).dataset.cap);
      if (this.#capabilities[cap]) {
        item.classList.add('supported');
        item.classList.remove('unsupported');
      } else {
        item.classList.add('unsupported');
        item.classList.remove('supported');
      }
    });
  }

  /**
   * Resolve GPU device name from adapter info with fallback chain
   * @param {GPUAdapterInfo} info
   * @returns {string}
   */
  #resolveGPUName(info) {
    const vendor = (info.vendor || '').toLowerCase();
    const device = (info.device || '').toLowerCase();
    const arch = (info.architecture || '').toLowerCase();

    // 1. Try parsing architecture string (works well on Apple Silicon)
    if (arch) {
      // Match patterns like "apple-m1", "apple-m2-pro", "apple-m3-max"
      const appleMatch = arch.match(/apple[- ]?(m\d+)(?:[- ]?(pro|max|ultra))?/i);
      if (appleMatch) {
        const chip = appleMatch[1].toUpperCase(); // M1, M2, M3, M4
        const variant = appleMatch[2]
          ? ` ${appleMatch[2].charAt(0).toUpperCase() + appleMatch[2].slice(1)}`
          : '';
        return `Apple ${chip}${variant}`;
      }
      // Return capitalized architecture if it looks meaningful
      if (arch.length > 3 && !arch.startsWith('0x')) {
        return arch.split('-').map(s => s.charAt(0).toUpperCase() + s.slice(1)).join(' ');
      }
    }

    // 2. Try description field
    if (info.description && info.description.length > 3) {
      return info.description;
    }

    // 3. Last resort: vendor + device (log for future mapping)
    if (vendor && device) {
      log.info('GPU', `Unknown device: vendor=${vendor}, device=${device}, arch=${arch}`);
      // Capitalize vendor
      const vendorName = vendor.charAt(0).toUpperCase() + vendor.slice(1);
      return `${vendorName} GPU`;
    }

    return 'Unknown GPU';
  }

  /**
   * Populate GPU info panel with adapter details
   * @param {GPUAdapter} adapter
   * @param {GPUAdapterInfo} info
   */
  #populateGPUInfo(adapter, info) {
    if (!this.#gpuElements.device) return;

    // Device name with friendly resolution
    const deviceName = this.#resolveGPUName(info);
    this.#gpuElements.device.textContent = deviceName;
    this.#gpuElements.device.title = deviceName;

    // Detect unified memory architecture (Apple Silicon, etc)
    const isUnifiedMemory = this.#isUnifiedMemoryArchitecture(info);
    this.#isUnifiedMemory = isUnifiedMemory;

    // System RAM (for unified memory systems)
    // Note: navigator.deviceMemory caps at 8GB for privacy
    // Override with ?ram=24 URL param to show actual RAM
    const urlParams = new URLSearchParams(window.location.search);
    const ramOverride = urlParams.get('ram');
    const deviceMemoryGB = /** @type {Navigator & { deviceMemory?: number }} */ (navigator).deviceMemory;
    const GB = 1024 * 1024 * 1024;
    if (isUnifiedMemory && this.#gpuElements.ramRow && this.#gpuElements.ram) {
      if (ramOverride) {
        const parsedOverride = Number.parseFloat(ramOverride);
        if (Number.isFinite(parsedOverride)) {
          this.#estimatedSystemMemoryBytes = parsedOverride * GB;
        }
        this.#gpuElements.ram.textContent = `${ramOverride} GB`;
      } else if (deviceMemoryGB && deviceMemoryGB >= 8) {
        // 8GB is the max reported, actual RAM is likely higher
        this.#estimatedSystemMemoryBytes = deviceMemoryGB * GB;
        this.#gpuElements.ram.textContent = '8+ GB';
      } else if (deviceMemoryGB) {
        this.#estimatedSystemMemoryBytes = deviceMemoryGB * GB;
        this.#gpuElements.ram.textContent = `${deviceMemoryGB} GB`;
      } else {
        this.#gpuElements.ram.textContent = 'Unknown';
      }
      this.#gpuElements.ramRow.hidden = false;
    }

    // Buffer limit (WebGPU's max buffer size - conservative on unified memory)
    const limits = /** @type {GPUSupportedLimits & { maxBufferSize?: number; maxStorageBufferBindingSize?: number }} */ (adapter.limits || {});
    const maxBufferSize = limits.maxBufferSize || 0;
    const maxStorageSize = limits.maxStorageBufferBindingSize || 0;
    const bufferLimit = Math.max(maxBufferSize, maxStorageSize);
    this.#gpuBufferLimitBytes = bufferLimit > 0 ? bufferLimit : null;
    if (bufferLimit > 0) {
      /** @type {HTMLElement} */ (this.#gpuElements.vram).textContent = this.#formatBytes(bufferLimit);
    } else {
      /** @type {HTMLElement} */ (this.#gpuElements.vram).textContent = 'Unknown';
    }

    // Show unified memory note
    if (isUnifiedMemory && this.#gpuElements.unifiedNote) {
      this.#gpuElements.unifiedNote.hidden = false;
    }

    // Features
    /** @type {string[]} */
    const features = [];
    if (this.#capabilities.f16) features.push('F16');
    if (this.#capabilities.subgroups) features.push('Subgroups');
    if (adapter.features.has('timestamp-query')) features.push('Timestamps');
    /** @type {HTMLElement} */ (this.#gpuElements.features).textContent = features.length > 0 ? features.join(', ') : 'Basic';
  }

  /**
   * Detect unified memory architecture (Apple Silicon, integrated GPUs)
   * @param {GPUAdapterInfo} info
   * @returns {boolean}
   */
  #isUnifiedMemoryArchitecture(info) {
    const arch = info.architecture?.toLowerCase() || '';
    const vendor = info.vendor?.toLowerCase() || '';
    const desc = info.description?.toLowerCase() || '';

    // Apple Silicon (M1/M2/M3/M4) uses unified memory
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
      // MacIntel on ARM is Apple Silicon via Rosetta or native
      // Modern Macs with M-series report various things, but if we see Metal, it's unified
      if (desc.includes('metal') || vendor.includes('apple')) {
        return true;
      }
    }

    return false;
  }

  /**
   * Start polling memory stats
   */
  #startMemoryPolling() {
    // Update immediately
    this.#updateMemoryStats();

    // Poll every 2 seconds
    this.#memoryPollInterval = setInterval(() => {
      this.#updateMemoryStats();
    }, 2000);
  }

  /**
   * Update memory stats display
   */
  #updateMemoryStats() {
    if (!this.#memoryElements.heapBar) return;

    let usedHeap = 0;
    let usedGpuPool = 0;
    let usedKv = 0;
    let totalLimit = 0;

    // JS Heap (from performance.memory if available - Chrome only)
    const memory = /** @type {Performance & { memory?: { usedJSHeapSize?: number; jsHeapSizeLimit?: number; totalJSHeapSize?: number } }} */ (performance).memory;
    if (memory) {
      usedHeap = memory.usedJSHeapSize || 0;
      const totalHeap = memory.jsHeapSizeLimit || memory.totalJSHeapSize || 1;
      const heapPercent = Math.min(100, (usedHeap / totalHeap) * 100);
      totalLimit = totalHeap;

      this.#memoryElements.heapBar.style.width = `${heapPercent}%`;
      /** @type {HTMLElement} */ (this.#memoryElements.heapValue).textContent = this.#formatBytes(usedHeap);
    } else {
      /** @type {HTMLElement} */ (this.#memoryElements.heapValue).textContent = 'N/A';
    }

    // GPU buffer usage (from buffer pool)
    /** @type {{ peakBytesAllocated?: number; currentBytesAllocated?: number } | null} */
    let poolStats = null;
    try {
      const bufferPool = getBufferPool();
      poolStats = bufferPool.getStats();
      usedGpuPool = poolStats.currentBytesAllocated || 0;
    } catch {
      /** @type {HTMLElement} */ (this.#memoryElements.gpuValue).textContent = '--';
    }

    // KV cache usage (if pipeline exposes it)
    if (this.#pipeline && typeof /** @type {any} */ (this.#pipeline).getMemoryStats === 'function') {
      const memStats = /** @type {any} */ (this.#pipeline).getMemoryStats();
      if (memStats?.kvCache?.allocated) {
        usedKv = memStats.kvCache.allocated;
      }
    }

    const usedGpuTotal = usedGpuPool + usedKv;
    const minGpuLimit = 4 * 1024 * 1024 * 1024;
    const peakBytes = poolStats?.peakBytesAllocated || 0;
    const gpuLimit = this.#gpuBufferLimitBytes || Math.max(peakBytes, usedGpuTotal, minGpuLimit);
    const gpuPercent = Math.min(100, (usedGpuTotal / gpuLimit) * 100);

    if (this.#memoryElements.gpuBar && this.#memoryElements.gpuValue) {
      this.#memoryElements.gpuBar.style.width = `${gpuPercent}%`;
      this.#memoryElements.gpuValue.textContent = this.#formatBytes(usedGpuTotal);
    }

    if (this.#memoryElements.kvBar && this.#memoryElements.kvValue) {
      const kvPercent = Math.min(100, (usedKv / gpuLimit) * 100);
      this.#memoryElements.kvBar.style.width = `${kvPercent}%`;
      this.#memoryElements.kvValue.textContent = usedKv > 0 ? this.#formatBytes(usedKv) : '--';
    }

    // Use GPU limit for total if larger than heap limit
    if (gpuLimit > totalLimit) totalLimit = gpuLimit;

    // OPFS cache storage (async, but we'll update on next cycle)
    if (this.#memoryElements.opfsBar && this.#memoryElements.opfsValue) {
      navigator.storage.estimate().then((estimate) => {
        const opfsUsed = estimate.usage || 0;
        const opfsQuota = estimate.quota || 1;
        const opfsPercent = Math.min(100, (opfsUsed / opfsQuota) * 100);

        /** @type {HTMLElement} */ (this.#memoryElements.opfsBar).style.width = `${opfsPercent}%`;
        /** @type {HTMLElement} */ (this.#memoryElements.opfsValue).textContent = this.#formatBytes(opfsUsed);
      }).catch(() => {
        /** @type {HTMLElement} */ (this.#memoryElements.opfsValue).textContent = '--';
      });
    }

    // Update stacked total bar
    if (this.#memoryElements.heapStackedBar && this.#memoryElements.gpuStackedBar) {
      const totalUsed = usedHeap + usedGpuTotal;
      // Calculate percentages relative to combined limit (or reasonable max)
      const combinedLimit = Math.max(totalLimit, 8 * 1024 * 1024 * 1024); // At least 8GB scale
      const heapStackedPercent = Math.min(50, (usedHeap / combinedLimit) * 100);
      const gpuStackedPercent = Math.min(50, (usedGpuTotal / combinedLimit) * 100);

      this.#memoryElements.heapStackedBar.style.width = `${heapStackedPercent}%`;
      this.#memoryElements.gpuStackedBar.style.width = `${gpuStackedPercent}%`;

      if (this.#memoryElements.totalValue) {
        this.#memoryElements.totalValue.textContent = this.#formatBytes(totalUsed);
      }

      // Detect potential swapping (total > device memory)
      // navigator.deviceMemory gives RAM in GB (approximate)
      const deviceMemoryGB = /** @type {Navigator & { deviceMemory?: number }} */ (navigator).deviceMemory;
      if (deviceMemoryGB && this.#swapIndicator) {
        const physicalRamBytes = deviceMemoryGB * 1024 * 1024 * 1024;
        // Show swap warning if we're using more than 90% of reported device memory
        // (deviceMemory is capped/rounded, so actual RAM may be higher)
        if (totalUsed > physicalRamBytes * 0.9) {
          this.#swapIndicator.hidden = false;
        } else {
          this.#swapIndicator.hidden = true;
        }
      }

      if (this.#memoryElements.headroomValue && this.#memoryElements.headroomBar) {
        const estimatedTotal = this.#getEstimatedSystemMemoryBytes();
        const headroomBytes = this.#estimateHeadroomBytes(totalUsed, estimatedTotal);
        if (headroomBytes === null || !estimatedTotal) {
          this.#memoryElements.headroomValue.textContent = '--';
          this.#memoryElements.headroomBar.style.width = '0%';
        } else {
          const headroomPercent = Math.min(100, (headroomBytes / estimatedTotal) * 100);
          this.#memoryElements.headroomValue.textContent = this.#formatBytes(headroomBytes);
          this.#memoryElements.headroomBar.style.width = `${headroomPercent}%`;
        }
      }
    }
  }

  /**
   * @returns {number | null}
   */
  #getEstimatedSystemMemoryBytes() {
    if (this.#estimatedSystemMemoryBytes) {
      return this.#estimatedSystemMemoryBytes;
    }
    const estimatedGB = this.#memoryCapabilities?.unifiedMemoryInfo?.estimatedMemoryGB;
    return estimatedGB ? estimatedGB * 1024 * 1024 * 1024 : null;
  }

  /**
   * @param {number} totalUsedBytes
   * @param {number | null} estimatedTotalBytes
   * @returns {number | null}
   */
  #estimateHeadroomBytes(totalUsedBytes, estimatedTotalBytes) {
    if (!this.#isUnifiedMemory || !estimatedTotalBytes) {
      return null;
    }
    return Math.max(0, estimatedTotalBytes - totalUsedBytes);
  }

  /**
   * Format bytes to human-readable string
   * @param {number} bytes
   * @returns {string}
   */
  #formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  /**
   * Format model ID to a human-readable name
   * @param {string} modelId
   * @returns {string}
   */
  #formatModelName(modelId) {
    // Remove common prefixes
    let name = modelId
      .replace(/^custom-\d+$/, 'Custom Model')
      .replace(/^tools\//, '')
      .replace(/-rdrr$/, '')
      .replace(/-q4$/, '')
      .replace(/-q4_k_m$/i, '');

    // If it looks like a timestamp-based ID, just call it "Custom Model"
    if (/^custom-\d+$/.test(modelId)) {
      return 'Custom Model';
    }

    // Title case the remaining parts
    return name
      .split(/[-_]/)
      .map((s) => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase())
      .join(' ');
  }

  /**
   * Generate a deduplication key for a model
   * Uses architecture + quantization only (size varies between sources)
   * @param {string | undefined} arch
   * @param {string | undefined} quant
   * @param {number | string} [_size]
   * @returns {string}
   */
  #getModelKey(arch, quant, _size) {
    // Normalize architecture: extract base model family (gemma, llama, mistral, etc.)
    const normArch = (arch || 'unknown')
      .toLowerCase()
      .replace(/forcausallm|forconditionalgeneration|model/gi, '')
      .replace(/[^a-z0-9]/g, '');

    // Normalize quantization
    const normQuant = (quant || 'unknown').toLowerCase().replace(/[^a-z0-9]/g, '');

    // Size intentionally excluded - varies between manifest and estimates
    return `${normArch}:${normQuant}`;
  }

  /**
   * Load list of cached models from storage, deduplicating by model identity
   * @returns {Promise<void>}
   */
  async #loadCachedModels() {
    log.debug('App', 'Discovering models...');

    // Map to deduplicate models: key -> model info with sources
    /** @type {Map<string, import('./app.js').RegisteredModel>} */
    const modelMap = new Map();

    /**
     * Helper to add/merge a model into the map
     * @param {string} key
     * @param {Partial<import('./app.js').RegisteredModel>} info
     * @param {keyof import('./model-selector.js').ModelSources} sourceType
     * @param {{ id: string; url?: string }} sourceData
     */
    const addModel = (key, info, sourceType, sourceData) => {
      if (modelMap.has(key)) {
        // Merge sources
        const existing = /** @type {import('./app.js').RegisteredModel} */ (modelMap.get(key));
        /** @type {Record<string, unknown>} */ (existing.sources)[sourceType] = sourceData;
        // Prefer better metadata (server > browser > remote)
        if (sourceType === 'server' || (sourceType === 'browser' && !existing.sources.server)) {
          existing.name = info.name || existing.name;
          existing.size = info.size || existing.size;
          existing.downloadSize = info.downloadSize || existing.downloadSize;
        }
      } else {
        modelMap.set(key, /** @type {import('./app.js').RegisteredModel} */ ({
          ...info,
          key,
          sources: { [sourceType]: sourceData },
        }));
      }
    };

    // 1. Discover server models (local HTTP)
    const serverModels = await discoverLocalModels();
    log.debug('App', `Found ${serverModels.length} server models`);

    for (const model of serverModels) {
      const key = this.#getModelKey(model.architecture, model.quantization, model.downloadSize);
      addModel(
        key,
        {
          name: model.name,
          size: model.size,
          quantization: model.quantization,
          downloadSize: model.downloadSize,
          architecture: model.architecture,
        },
        'server',
        { id: model.id, url: model.url }
      );
    }

    // 2. Check OPFS for browser-cached models
    /** @type {string[]} */
    let cachedIds = [];
    try {
      cachedIds = await listModels();
      log.debug('App', 'Found cached models in OPFS:', cachedIds);
    } catch (err) {
      log.warn('App', 'Could not query cached models:', /** @type {Error} */ (err).message);
    }

    for (const cachedId of cachedIds) {
      try {
        await openModelDirectory(cachedId);
        const manifestText = await loadManifestFromOPFS();
        if (manifestText) {
          const manifest = parseManifest(manifestText);
          const archInfo = manifest.architecture;
          const archLabel = typeof archInfo === 'string' ? archInfo : manifest.modelType;
          const quant = manifest.quantization || 'Unknown';
          const totalSize = (manifest.shards || []).reduce((/** @type {number} */ sum, /** @type {{ size?: number }} */ s) => sum + (s.size || 0), 0);

          // Estimate param count
          const hiddenSize =
            typeof archInfo === 'object' && archInfo !== null ? /** @type {{ hiddenSize?: number }} */ (archInfo).hiddenSize || 0 : 0;
          let paramStr = 'Unknown';
          if (hiddenSize >= 4096) paramStr = '7B+';
          else if (hiddenSize >= 2048) paramStr = '1-3B';
          else if (hiddenSize >= 1024) paramStr = '<1B';

          const key = this.#getModelKey(archLabel || 'unknown', quant, totalSize);
          addModel(
            key,
            {
              name: manifest.modelId || this.#formatModelName(cachedId),
              architecture: archLabel || 'Unknown',
              size: paramStr,
              quantization: quant,
              downloadSize: totalSize,
            },
            'browser',
            { id: cachedId }
          );
        }
      } catch (e) {
        log.warn('App', `Could not load manifest for cached model ${cachedId}:`, /** @type {Error} */ (e).message);
      }
    }

    // 3. Add remote models (available for download)
    for (const remote of REMOTE_MODELS) {
      const key = this.#getModelKey(remote.architecture || remote.id, remote.quantization, remote.downloadSize);
      addModel(
        key,
        {
          name: remote.name,
          size: remote.size,
          quantization: remote.quantization,
          downloadSize: remote.downloadSize,
          architecture: remote.architecture,
        },
        'remote',
        { id: remote.id, url: remote.url }
      );
    }

    // 4. Add Quick Start models (CDN-hosted with preflight checks)
    for (const [modelId, config] of Object.entries(QUICKSTART_MODELS)) {
      const req = config.requirements;
      const key = this.#getModelKey(req.architecture || modelId, req.quantization, req.downloadSize);
      const existing = modelMap.get(key);
      if (existing) {
        // Mark existing model as quick-start available
        existing.quickStartAvailable = true;
      } else {
        // Add as new remote model with quick-start
        addModel(
          key,
          {
            name: config.displayName,
            size: req.paramCount,
            quantization: req.quantization,
            downloadSize: req.downloadSize,
            architecture: req.architecture,
            quickStartAvailable: true,
          },
          'remote',
          { id: modelId, url: config.baseUrl }
        );
      }
    }

    // 5. Convert map to array and sort by availability
    // Priority: server+browser > server > browser > remote
    /**
     * @param {import('./app.js').RegisteredModel} m
     * @returns {number}
     */
    const getAvailabilityScore = (m) => {
      let score = 0;
      if (m.sources.server) score += 2;
      if (m.sources.browser) score += 1;
      return score;
    };

    MODEL_REGISTRY = Array.from(modelMap.values()).sort((a, b) => {
      return getAvailabilityScore(b) - getAvailabilityScore(a);
    });

    log.info('App', `Model registry: ${MODEL_REGISTRY.length} unique models`);
    this.#modelSelector?.setModels(/** @type {import('./model-selector.js').ModelInfo[]} */ (MODEL_REGISTRY));
  }

  /**
   * Select and load a model (run it)
   * @param {import('./app.js').RegisteredModel | string} modelOrKey
   * @param {{ preferredSource?: string }} [opts]
   * @returns {Promise<void>}
   */
  async selectModel(modelOrKey, opts = {}) {
    if (this.#isGenerating) {
      this.#showError('Cannot switch models while generating');
      return;
    }

    // Support both model object and key string
    const model =
      typeof modelOrKey === 'string'
        ? MODEL_REGISTRY.find((m) => m.key === modelOrKey)
        : modelOrKey;

    if (!model) {
      this.#showError(`Unknown model: ${modelOrKey}`);
      return;
    }

    const sources = model.sources || {};
    const hasServer = !!sources.server;
    const hasBrowser = !!sources.browser;

    if (!hasServer && !hasBrowser) {
      this.#showError('Model not available locally. Download it first.');
      return;
    }

    // Use preferred source if specified, otherwise default to server > browser
    /** @type {boolean} */
    let useServer;
    if (opts.preferredSource === 'server' && hasServer) {
      useServer = true;
    } else if (opts.preferredSource === 'browser' && hasBrowser) {
      useServer = false;
    } else {
      useServer = hasServer; // Default: prefer server
    }

    const sourceInfo = useServer ? /** @type {NonNullable<typeof sources.server>} */ (sources.server) : /** @type {NonNullable<typeof sources.browser>} */ (sources.browser);
    const sourceType = useServer ? 'server' : 'browser';

    log.info('App', `Loading model: ${model.name} from ${sourceType}`);
    this.#setStatus('loading', 'Loading model...');
    this.#progressUI?.show('Loading model...');

    try {
      // Unload current model if any
      if (this.#pipeline) {
        if (typeof /** @type {any} */ (this.#pipeline).unload === 'function') {
          await /** @type {any} */ (this.#pipeline).unload();
        }
        this.#pipeline = null;
      }

      /** @type {import('../src/storage/rdrr-format.js').RDRRManifest} */
      let manifest;
      /** @type {(idx: number) => Promise<ArrayBuffer>} */
      let loadShardFn;

      // Track loading source for progress - distinguish between network/disk/cache
      const isLocalServer = sourceInfo.url?.match(/^(https?:\/\/)?(localhost|127\.0\.0\.1|0\.0\.0\.0|file:)/i);
      const loadSourceType = useServer ? (isLocalServer ? 'disk' : 'network') : 'cache';
      this.#progressUI?.setSourceType(/** @type {import('./progress-ui.js').SourceType} */ (loadSourceType));

      if (useServer) {
        // Load from HTTP - show source phase
        this.#progressUI?.setPhaseProgress({ phase: 'source', percent: 5, message: 'Fetching manifest...' });
        const manifestUrl = `${sourceInfo.url}/manifest.json`;
        const response = await fetch(manifestUrl);
        if (!response.ok) throw new Error(`Failed to fetch manifest: ${response.status}`);
        manifest = parseManifest(await response.text());

        // Create HTTP shard loader
        loadShardFn = async (idx) => {
          const shard = manifest.shards[idx];
          const shardUrl = `${sourceInfo.url}/${shard.filename}`;
          const res = await fetch(shardUrl);
          if (!res.ok) throw new Error(`Failed to fetch shard ${idx}: ${res.status}`);
          return await res.arrayBuffer();
        };
      } else {
        // Load from OPFS (browser cache) - show source phase (labeled "Cache")
        await openModelDirectory(sourceInfo.id);
        this.#progressUI?.setPhaseProgress({ phase: 'source', percent: 5, message: 'Loading manifest...' });
        const manifestJson = await loadManifestFromOPFS();
        manifest = parseManifest(manifestJson);

        // Create OPFS shard loader
        const { loadShard } = await import('../src/storage/shard-manager.js');
        loadShardFn = (idx) => loadShard(idx);
      }

      // Initialize GPU - show GPU phase starting
      this.#progressUI?.setPhaseProgress({ phase: 'gpu', percent: 5, message: 'Initializing...' });

      // Ensure GPU device is initialized
      const device = getDevice() || (await initDevice());
      const gpuCaps = getKernelCapabilities();
      const memCaps = await getMemoryCapabilities();
      this.#memoryCapabilities = memCaps;
      if (!this.#estimatedSystemMemoryBytes && memCaps.unifiedMemoryInfo.estimatedMemoryGB) {
        this.#estimatedSystemMemoryBytes = memCaps.unifiedMemoryInfo.estimatedMemoryGB * 1024 * 1024 * 1024;
      }
      if (memCaps.isUnifiedMemory) {
        this.#isUnifiedMemory = true;
      }
      const heapManager = getHeapManager();
      await heapManager.init();

      this.#progressUI?.setPhaseProgress({ phase: 'gpu', percent: 10, message: 'Creating pipeline...' });

      // Create pipeline with multi-phase progress tracking
      this.#pipeline = await createPipeline(/** @type {any} */ (manifest), {
        gpu: {
          capabilities: gpuCaps,
          device: device,
        },
        memory: {
          capabilities: memCaps,
          heapManager: heapManager,
        },
        storage: {
          loadShard: loadShardFn,
        },
        baseUrl: useServer ? sourceInfo.url : undefined,
        runtime: {
          debug: new URLSearchParams(window.location.search).has('debug'),
          kernelPath: this.#runtimeKernelPath || undefined,
        },
        onProgress: (/** @type {{ percent: number; message?: string; stage?: string; layer?: number; total?: number; bytesLoaded?: number; totalBytes?: number; bytesPerSecond?: number }} */ progress) => {
          const stage = progress.stage || 'layers';

          // Map loader stages to UI phases (simplified: source + gpu)
          if (stage === 'manifest' || stage === 'shards') {
            // Shard loading from source
            this.#progressUI?.setPhaseProgress({
              phase: 'source',
              percent: Math.min(100, progress.percent * 1.2), // Scale to show some progress
              bytesLoaded: progress.bytesLoaded,
              totalBytes: progress.totalBytes,
              speed: progress.bytesPerSecond,
            });
          } else if (stage === 'layers' || stage === 'gpu_transfer') {
            // Mark source complete, show GPU progress
            this.#progressUI?.setPhaseProgress({ phase: 'source', percent: 100, message: 'Complete' });

            // GPU phase: show layer progress
            const gpuPercent = 10 + (progress.percent * 0.9);
            /** @type {string} */
            let message;
            if (progress.layer !== undefined && progress.total) {
              message = `Layer ${progress.layer}/${progress.total}`;
            } else if (stage === 'gpu_transfer') {
              message = 'Uploading weights...';
            } else {
              message = `${Math.round(gpuPercent)}%`;
            }
            this.#progressUI?.setPhaseProgress({
              phase: 'gpu',
              percent: gpuPercent,
              message,
            });
          } else if (stage === 'complete') {
            // All phases complete
            this.#progressUI?.setPhaseProgress({ phase: 'source', percent: 100, message: 'Done' });
            this.#progressUI?.setPhaseProgress({ phase: 'gpu', percent: 100, message: 'Ready' });
          }
        },
      });

      this.#currentModel = model;
      this.#modelSelector?.setActiveModel(model.key);
      this.#progressUI?.hide();
      this.#setStatus('ready', `${model.name} loaded`);
      this.#chatUI?.setInputEnabled(true);
      this.#chatUI?.focusInput();
      this.#updateInitialStats();

      log.info('App', `Model loaded: ${model.name} (${model.key})`);
    } catch (error) {
      log.error('App', 'Model load failed:', error);
      this.#progressUI?.hide();
      this.#setStatus('error', 'Load failed');
      this.#showError(`Failed to load model: ${/** @type {Error} */ (error).message}`);
    }
  }

  /**
   * Download/cache a model to browser storage
   * @param {import('./app.js').RegisteredModel} model
   * @param {{ runAfter?: boolean }} [opts]
   * @returns {Promise<void>}
   */
  async downloadModel(model, opts = {}) {
    const sources = model.sources || {};

    // Determine URL: prefer server (for caching), then remote
    /** @type {string | null} */
    let downloadUrl = null;
    let storageId = model.key.replace(/[^a-zA-Z0-9_-]/g, '_'); // Safe filename

    if (sources.server) {
      downloadUrl = /** @type {string} */ (sources.server.url);
    } else if (sources.remote) {
      downloadUrl = /** @type {string} */ (sources.remote.url);
      storageId = sources.remote.id || storageId;
    }

    if (!downloadUrl) {
      this.#showError('No download source available');
      return;
    }

    log.info('App', `Downloading "${model.name}" from: ${downloadUrl}`);
    this.#setStatus('loading', `Downloading ${model.name}...`);

    try {
      const success = await downloadModel(
        downloadUrl,
        (/** @type {import('../src/storage/downloader.js').DownloadProgress} */ progress) => {
          const percent =
            progress.totalBytes > 0
              ? Math.round((progress.downloadedBytes / progress.totalBytes) * 100)
              : 0;
          this.#modelSelector?.setDownloadProgress(model.key, percent);

          if (progress.stage === 'verifying') {
            this.#setStatus('loading', 'Verifying...');
          }
        },
        { modelId: storageId }
      );

      if (!success) {
        throw new Error('Download failed');
      }

      this.#setStatus('ready', 'Download complete');

      // Refresh models list to update sources
      await this.#loadCachedModels();

      log.info('App', `Download complete: ${model.name}`);

      // Run after download if requested
      if (opts.runAfter) {
        // Find the updated model in registry
        const updatedModel = MODEL_REGISTRY.find((m) => m.key === model.key);
        if (updatedModel) {
          await this.selectModel(updatedModel);
        }
      }
    } catch (error) {
      log.error('App', 'Download failed:', error);
      this.#modelSelector?.setDownloadProgress(model.key, 0);
      this.#setStatus('error', 'Download failed');
      this.#showError(`Download failed: ${/** @type {Error} */ (error).message}`);
    }
  }

  /**
   * Delete a model from browser cache
   * @param {import('./app.js').RegisteredModel} model
   * @returns {Promise<void>}
   */
  async deleteModel(model) {
    const sources = model.sources || {};
    const browserId = sources.browser?.id;

    if (!browserId) {
      this.#showError('Model is not cached in browser');
      return;
    }

    log.info('App', `Deleting cached model: ${model.name} (${browserId})`);

    try {
      // Unload if currently active
      if (this.#currentModel?.key === model.key) {
        if (this.#pipeline) {
          if (typeof /** @type {any} */ (this.#pipeline).unload === 'function') {
            await /** @type {any} */ (this.#pipeline).unload();
          }
          this.#pipeline = null;
        }
        this.#currentModel = null;
        this.#modelSelector?.setActiveModel(null);
        this.#chatUI?.setInputEnabled(false);
      }

      // Delete from OPFS
      await deleteModelFromOPFS(browserId);
      this.#setStatus('ready', 'Cache cleared');

      // Refresh models list
      await this.#loadCachedModels();
    } catch (error) {
      log.error('App', 'Delete failed:', error);
      this.#showError(`Delete failed: ${/** @type {Error} */ (error).message}`);
    }
  }

  /**
   * Send a chat message and generate response
   * @param {string} message
   * @returns {Promise<void>}
   */
  async chat(message) {
    if (!this.#currentModel) {
      this.#showError('No model loaded');
      return;
    }

    if (!this.#pipeline) {
      this.#showError('Pipeline not initialized');
      return;
    }

    if (this.#isGenerating) {
      return;
    }

    log.debug('App', 'Generating response...');
    this.#isGenerating = true;
    this.#abortController = new AbortController();

    // Add user message
    this.#chatUI?.addMessage('user', message);

    // Start streaming response
    this.#chatUI?.startStream();
    this.#setStatus('loading', 'Generating...');

    try {
      // Use real pipeline generation
      let tokenCount = 0;
      const startTime = performance.now();

      for await (const token of this.#pipeline.generate(message, {
        maxTokens: 512,
        temperature: this.#getSamplingTemperature(),
        topP: this.#getSamplingTopP(),
        topK: this.#getSamplingTopK(),
        signal: this.#abortController.signal,
      })) {
        if (this.#abortController.signal.aborted) break;
        this.#chatUI?.streamToken(token);
        tokenCount++;

        // Update TPS periodically
        if (tokenCount % 10 === 0) {
          const elapsed = (performance.now() - startTime) / 1000;
          this.#updateStats(tokenCount / elapsed);
        }
      }

      const stats = this.#chatUI?.finishStream();
      if (stats) {
        this.#updateStats(stats.tokensPerSec);
      }
      this.#setStatus('ready', `${this.#currentModel.name}`);
    } catch (error) {
      if (/** @type {Error} */ (error).name === 'AbortError') {
        this.#chatUI?.cancelStream();
        this.#setStatus('ready', 'Stopped');
      } else {
        log.error('App', 'Generation error:', error);
        this.#chatUI?.cancelStream();
        this.#setStatus('error', 'Generation failed');
        this.#showError(`Generation failed: ${/** @type {Error} */ (error).message}`);
      }
    } finally {
      this.#isGenerating = false;
      this.#abortController = null;
    }
  }

  /**
   * @returns {number}
   */
  #getSamplingTemperature() {
    const n = parseFloat(this.#temperatureInput?.value || '');
    return Number.isFinite(n) ? n : 0.7;
  }

  /**
   * @returns {number}
   */
  #getSamplingTopP() {
    const n = parseFloat(this.#topPInput?.value || '');
    return Number.isFinite(n) ? n : 0.9;
  }

  /**
   * @returns {number}
   */
  #getSamplingTopK() {
    const n = parseInt(this.#topKInput?.value || '', 10);
    return Number.isFinite(n) ? n : 40;
  }

  /**
   * Stop current generation
   */
  stopGeneration() {
    if (this.#abortController) {
      this.#abortController.abort();
    }
  }

  /**
   * Clear conversation history
   */
  clearConversation() {
    if (this.#pipeline && typeof /** @type {any} */ (this.#pipeline).clearKVCache === 'function') {
      /** @type {any} */ (this.#pipeline).clearKVCache();
    }
    this.#chatUI?.clear();
    log.debug('App', 'Conversation cleared');
  }

  /**
   * Get current status
   * @returns {{ model: string | null; modelName: string | null; isGenerating: boolean; capabilities: import('./app.js').Capabilities; memory: unknown; gpu: unknown }}
   */
  getStatus() {
    /** @type {unknown} */
    let memoryUsage = null;
    /** @type {unknown} */
    let gpuUsage = null;

    if (this.#pipeline) {
      // Get memory stats from pipeline if available
      if (typeof /** @type {any} */ (this.#pipeline).getMemoryStats === 'function') {
        memoryUsage = /** @type {any} */ (this.#pipeline).getMemoryStats();
      }
      // Get GPU stats if available
      if (typeof /** @type {any} */ (this.#pipeline).getGPUStats === 'function') {
        gpuUsage = /** @type {any} */ (this.#pipeline).getGPUStats();
      }
    }

    return {
      model: this.#currentModel?.key || null,
      modelName: this.#currentModel?.name || null,
      isGenerating: this.#isGenerating,
      capabilities: { ...this.#capabilities },
      memory: memoryUsage,
      gpu: gpuUsage,
    };
  }

  /**
   * Set status indicator
   * @param {string} state
   * @param {string} text
   */
  #setStatus(state, text) {
    if (this.#statusDot) {
      this.#statusDot.className = `status-dot ${state}`;
    }
    if (this.#statusText) {
      this.#statusText.textContent = text;
    }
  }

  /**
   * Update performance stats
   * @param {number} tps
   */
  #updateStats(tps) {
    if (this.#statsElements.tps) {
      this.#statsElements.tps.textContent = tps.toFixed(1);
    }

    // Update memory and GPU stats from pipeline
    if (this.#pipeline) {
      if (this.#statsElements.memory && typeof /** @type {any} */ (this.#pipeline).getMemoryStats === 'function') {
        const memStats = /** @type {any} */ (this.#pipeline).getMemoryStats();
        if (memStats && memStats.used) {
          const usedMB = (memStats.used / 1024 / 1024).toFixed(0);
          this.#statsElements.memory.textContent = `${usedMB} MB`;
        }
      }

      if (this.#statsElements.kv && typeof /** @type {any} */ (this.#pipeline).getKVCacheStats === 'function') {
        const kvStats = /** @type {any} */ (this.#pipeline).getKVCacheStats();
        if (kvStats) {
          this.#statsElements.kv.textContent = `${kvStats.seqLen}/${kvStats.maxSeqLen}`;
        }
      }
    }
  }

  /**
   * Update initial stats after model load (before generation starts)
   */
  #updateInitialStats() {
    // Show TPS as "Ready" before generation starts
    if (this.#statsElements.tps) {
      this.#statsElements.tps.textContent = 'Ready';
    }

    // Show memory usage
    if (this.#statsElements.memory) {
      const heap = /** @type {Performance & { memory?: { usedJSHeapSize: number } }} */ (performance).memory?.usedJSHeapSize;
      if (heap) {
        const usedMB = (heap / 1024 / 1024).toFixed(0);
        this.#statsElements.memory.textContent = `${usedMB} MB`;
      }
    }

    // Show GPU buffer count if available
    if (this.#statsElements.gpu && this.#pipeline) {
      const bufferPool = /** @type {any} */ (this.#pipeline).getBufferPool?.();
      if (bufferPool && typeof bufferPool.getStats === 'function') {
        const stats = bufferPool.getStats();
        this.#statsElements.gpu.textContent = `${stats.activeBuffers}`;
      } else {
        this.#statsElements.gpu.textContent = 'N/A';
      }
    }

    // Show KV cache as "0/max" initially
    if (this.#statsElements.kv && this.#pipeline) {
      const kvStats = /** @type {any} */ (this.#pipeline).getKVCacheStats?.();
      if (kvStats) {
        this.#statsElements.kv.textContent = `${kvStats.seqLen}/${kvStats.maxSeqLen}`;
      } else {
        this.#statsElements.kv.textContent = '0';
      }
    }

    // Enable unload button
    if (this.#unloadModelBtn) {
      this.#unloadModelBtn.disabled = false;
    }
  }

  /**
   * Unload the current model from GPU memory
   * @returns {Promise<void>}
   */
  async #unloadCurrentModel() {
    if (!this.#pipeline) {
      log.debug('App', 'No model loaded');
      return;
    }

    log.info('App', 'Unloading current model...');
    this.#setStatus('loading', 'Unloading model...');

    try {
      // Call pipeline unload if available
      if (typeof /** @type {any} */ (this.#pipeline).unload === 'function') {
        await /** @type {any} */ (this.#pipeline).unload();
      }

      this.#pipeline = null;
      this.#currentModel = null;
      this.#modelSelector?.setActiveModel(null);
      this.#chatUI?.setInputEnabled(false);

      // Reset stats
      if (this.#statsElements.tps) this.#statsElements.tps.textContent = '--';
      if (this.#statsElements.memory) this.#statsElements.memory.textContent = '--';
      if (this.#statsElements.gpu) this.#statsElements.gpu.textContent = '--';
      if (this.#statsElements.kv) this.#statsElements.kv.textContent = '--';

      // Disable unload button
      if (this.#unloadModelBtn) {
        this.#unloadModelBtn.disabled = true;
      }

      this.#setStatus('ready', 'Model unloaded');
      this.#updateMemoryStats();
      log.info('App', 'Model unloaded');
    } catch (error) {
      log.error('App', 'Unload failed:', error);
      this.#setStatus('error', 'Unload failed');
    }
  }

  /**
   * Clear all GPU memory (buffers, caches, heap)
   * @returns {Promise<void>}
   */
  async #clearAllMemory() {
    log.info('App', 'Clearing all memory...');
    this.#setStatus('loading', 'Clearing memory...');

    try {
      // First unload the model if loaded
      if (this.#pipeline) {
        await this.#unloadCurrentModel();
      }

      // Destroy buffer pool
      const { destroyBufferPool } = await import('../src/gpu/buffer-pool.js');
      destroyBufferPool();

      // Reset heap manager
      const { getHeapManager } = await import('../src/memory/heap-manager.js');
      const heapManager = getHeapManager();
      if (heapManager && typeof heapManager.reset === 'function') {
        heapManager.reset();
      }

      // Force garbage collection hint (may or may not work)
      if (typeof /** @type {typeof globalThis & { gc?: () => void }} */ (globalThis).gc === 'function') {
        /** @type {typeof globalThis & { gc: () => void }} */ (globalThis).gc();
      }

      this.#setStatus('ready', 'Memory cleared');
      this.#updateMemoryStats();
      log.info('App', 'All memory cleared');
    } catch (error) {
      log.error('App', 'Clear memory failed:', error);
      this.#setStatus('error', 'Clear failed');
    }
  }

  /**
   * Show error modal
   * @param {string} message
   */
  #showError(message) {
    const modal = /** @type {HTMLElement | null} */ (document.querySelector('#error-modal'));
    const messageEl = /** @type {HTMLElement | null} */ (document.querySelector('#error-message'));
    const closeBtn = /** @type {HTMLElement | null} */ (document.querySelector('#error-close'));

    if (messageEl) {
      messageEl.textContent = message;
    }
    if (modal) {
      modal.hidden = false;
    }

    const close = () => {
      if (modal) {
        modal.hidden = true;
      }
      closeBtn?.removeEventListener('click', close);
    };
    closeBtn?.addEventListener('click', close);
  }

  /**
   * Handle model conversion
   * @returns {Promise<void>}
   */
  async #handleConvert() {
    if (this.#isConverting) {
      return;
    }

    try {
      // Pick files
      const files = await pickModelFiles();
      if (!files || files.length === 0) {
        return;
      }

      log.info('App', `Converting ${files.length} files...`);
      this.#isConverting = true;
      if (this.#convertBtn) {
        this.#convertBtn.disabled = true;
      }

      // Show progress UI
      if (this.#convertStatus) {
        this.#convertStatus.hidden = false;
      }
      this.#updateConvertProgress(0, 'Starting conversion...');

      // Convert model
      const modelId = await convertModel(files, {
        onProgress: (/** @type {import('../src/browser/browser-converter.js').ConvertProgress} */ progress) => {
          const percent = progress.percent || 0;
          const message = progress.message || progress.stage;
          this.#updateConvertProgress(percent, message);

          if (progress.stage === ConvertStage.ERROR) {
            throw new Error(progress.message);
          }
        },
      });

      log.info('App', `Conversion complete: ${modelId}`);
      this.#updateConvertProgress(100, `Done! Model: ${modelId}`);

      // Refresh model list
      await this.#loadCachedModels();

      // Hide progress after delay
      setTimeout(() => {
        if (this.#convertStatus) {
          this.#convertStatus.hidden = true;
        }
        this.#updateConvertProgress(0, 'Ready');
      }, 3000);
    } catch (error) {
      if (/** @type {Error} */ (error).name === 'AbortError') {
        log.info('App', 'Conversion cancelled');
        this.#updateConvertProgress(0, 'Cancelled');
      } else {
        log.error('App', 'Conversion failed:', error);
        this.#updateConvertProgress(0, `Error: ${/** @type {Error} */ (error).message}`);
        this.#showError(`Conversion failed: ${/** @type {Error} */ (error).message}`);
      }
    } finally {
      this.#isConverting = false;
      if (this.#convertBtn) {
        this.#convertBtn.disabled = false;
      }
    }
  }

  /**
   * Update conversion progress UI
   * @param {number} percent
   * @param {string} message
   */
  #updateConvertProgress(percent, message) {
    if (this.#convertProgress) {
      this.#convertProgress.style.width = `${percent}%`;
    }
    if (this.#convertMessage) {
      this.#convertMessage.textContent = message;
    }
  }

  // ============================================================================
  // Quick-Start Methods
  // ============================================================================

  /**
   * Start quick-start flow for a model
   * @param {string} modelId
   * @returns {Promise<void>}
   */
  async startQuickStart(modelId) {
    const config = QUICKSTART_MODELS[modelId];
    if (!config) {
      this.#showError(`Unknown quick-start model: ${modelId}`);
      return;
    }

    log.info('QuickStart', `Starting download for ${modelId}`);

    const result = await downloadQuickStartModel(modelId, {
      onPreflightComplete: (preflight) => {
        log.debug('QuickStart', 'Preflight:', preflight);

        // Show VRAM blocker if needed
        if (!preflight.vram.sufficient) {
          this.#quickStartUI?.showVRAMBlocker(
            preflight.vram.required,
            preflight.vram.available
          );
        }
      },
      onStorageConsent: async (required, available, modelName) => {
        // Show consent dialog and wait for user response
        const consent = await this.#quickStartUI?.showStorageConsent(
          modelName,
          required,
          available
        );
        if (consent) {
          this.#quickStartUI?.showDownloadProgress();
        }
        return consent ?? false;
      },
      onProgress: (progress) => {
        this.#quickStartUI?.setDownloadProgress(
          progress.percent,
          progress.downloadedBytes,
          progress.totalBytes,
          progress.speed
        );
      },
    });

    if (result.success) {
      this.#quickStartUI?.showReady(modelId);
    } else if (result.blockedByPreflight) {
      // Already showing VRAM blocker
      log.debug('QuickStart', 'Blocked by preflight:', result.error);
    } else if (result.userDeclined) {
      log.debug('QuickStart', 'User declined');
      this.#quickStartUI?.hide();
    } else {
      this.#quickStartUI?.showError(result.error || 'Download failed');
    }
  }

  /**
   * Handle quick-start download completion
   * @param {string} modelId
   * @returns {Promise<void>}
   */
  async #onQuickStartComplete(modelId) {
    log.info('QuickStart', `Download complete for ${modelId}`);
    // Refresh model list to show the downloaded model
    await this.#loadCachedModels();
  }

  /**
   * Run model after quick-start download
   * @param {string} modelId
   * @returns {Promise<void>}
   */
  async #runQuickStartModel(modelId) {
    log.info('QuickStart', `Running model ${modelId}`);

    // Find the model in registry and select it
    const model = MODEL_REGISTRY.find((m) => m.key === modelId || m.sources.browser?.id === modelId);
    if (model) {
      await this.selectModel(model);
    } else {
      // Refresh and try again
      await this.#loadCachedModels();
      const refreshedModel = MODEL_REGISTRY.find((m) => m.key === modelId || m.sources.browser?.id === modelId);
      if (refreshedModel) {
        await this.selectModel(refreshedModel);
      } else {
        this.#showError(`Model ${modelId} not found after download`);
      }
    }
  }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new DopplerDemo();
  app.init().catch((err) => log.error('App', 'Initialization failed', err));

  // Expose for debugging
  /** @type {Window & { dopplerDemo?: DopplerDemo }} */ (window).dopplerDemo = app;
});

export default DopplerDemo;
