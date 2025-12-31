/**
 * app.ts - DOPPLER Application Controller
 *
 * Main application that wires together all components and the DOPPLER inference pipeline.
 *
 * @module app/app
 */

import { ModelSelector, ModelInfo, ModelSources } from './model-selector.js';
import { ChatUI } from './chat-ui.js';
import { ProgressUI } from './progress-ui.js';
import { QuickStartUI } from './quickstart-ui.js';

// Quick-start downloader
import {
  downloadQuickStartModel,
  QUICKSTART_MODELS,
  type QuickStartDownloadResult,
} from '../storage/quickstart-downloader.js';

// Browser model converter
import {
  convertModel,
  pickModelFiles,
  isConversionSupported,
  ConvertStage,
  ConvertProgress,
} from '../browser/model-converter.js';

// DOPPLER pipeline imports
import { createPipeline, Pipeline } from '../inference/pipeline.js';
import { downloadModel, DownloadProgress } from '../storage/downloader.js';
import {
  listModels,
  openModelDirectory,
  loadManifestFromOPFS,
  deleteModel as deleteModelFromOPFS,
} from '../storage/shard-manager.js';
import { parseManifest, RDRRManifest, type KernelHints } from '../storage/rdrr-format.js';
import { getMemoryCapabilities, MemoryCapabilities } from '../memory/capability.js';
import { getHeapManager, HeapManager } from '../memory/heap-manager.js';
import { getBufferPool } from '../gpu/buffer-pool.js';
import { initDevice, getKernelCapabilities, getDevice, KernelCapabilities } from '../gpu/device.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Remote model definition
 */
interface RemoteModel {
  id: string;
  name: string;
  size: string;
  quantization: string;
  downloadSize: number;
  url: string;
  source: string;
  downloaded?: boolean;
  architecture?: string;
}

/**
 * Capabilities state
 */
interface Capabilities {
  webgpu: boolean;
  f16: boolean;
  subgroups: boolean;
  memory64: boolean;
}

/**
 * Stats DOM elements
 */
interface StatsElements {
  tps: HTMLElement | null;
  memory: HTMLElement | null;
  gpu: HTMLElement | null;
  kv: HTMLElement | null;
}

/**
 * GPU info DOM elements
 */
interface GPUElements {
  device: HTMLElement | null;
  vram: HTMLElement | null;
  vramLabel: HTMLElement | null;
  ram: HTMLElement | null;
  ramRow: HTMLElement | null;
  features: HTMLElement | null;
  unifiedNote: HTMLElement | null;
}

/**
 * Memory bar DOM elements
 */
interface MemoryElements {
  heapBar: HTMLElement | null;
  heapValue: HTMLElement | null;
  gpuBar: HTMLElement | null;
  gpuValue: HTMLElement | null;
  opfsBar: HTMLElement | null;
  opfsValue: HTMLElement | null;
  // Stacked total bar
  heapStackedBar: HTMLElement | null;
  gpuStackedBar: HTMLElement | null;
  totalValue: HTMLElement | null;
}

/**
 * Registered model with sources
 */
interface RegisteredModel extends ModelInfo {
  key: string;
  sources: ModelSources;
}

/**
 * Server model from API
 */
interface ServerModel {
  name: string;
  path: string;
  numLayers?: number;
  vocabSize?: number;
  quantization?: string;
  downloadSize?: number;
  architecture?: string;
}

// ============================================================================
// Constants
// ============================================================================

/**
 * Remote models available for download
 * Currently empty - only local models (discovered via /api/models) are shown
 */
const REMOTE_MODELS: RemoteModel[] = [];

/** Dynamic model registry populated at runtime */
let MODEL_REGISTRY: RegisteredModel[] = [];

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Discover local models via server API
 */
async function discoverLocalModels(): Promise<RemoteModel[]> {
  const baseUrl = window.location.origin;

  try {
    const response = await fetch(`${baseUrl}/api/models`);
    if (!response.ok) return [];

    const models: ServerModel[] = await response.json();
    return models.map((m) => {
      // Create friendly name from folder name
      let modelName = m.name
        .replace(/-rdrr$/, '')
        .replace(/-q4$/, '')
        .split('-')
        .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
        .join(' ');

      // Infer param count from layers/hidden
      const inferredParams = m.numLayers && m.vocabSize ? `${m.numLayers}L` : 'Unknown';

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
    console.warn('[Discovery] Failed to fetch models from API:', e);
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
  private modelSelector: ModelSelector | null = null;
  private chatUI: ChatUI | null = null;
  private progressUI: ProgressUI | null = null;
  private quickStartUI: QuickStartUI | null = null;

  // Pipeline state
  private pipeline: Pipeline | null = null;
  private currentModel: RegisteredModel | null = null;
  private isGenerating = false;
  private abortController: AbortController | null = null;

  // Capabilities
  private capabilities: Capabilities = {
    webgpu: false,
    f16: false,
    subgroups: false,
    memory64: false,
  };

  // DOM references
  private statusDot: HTMLElement | null = null;
  private statusText: HTMLElement | null = null;
  private capabilitiesList: HTMLElement | null = null;
  private statsElements: StatsElements = {
    tps: null,
    memory: null,
    gpu: null,
    kv: null,
  };

  // GPU info elements
  private gpuElements: GPUElements = {
    device: null,
    vram: null,
    vramLabel: null,
    ram: null,
    ramRow: null,
    features: null,
    unifiedNote: null,
  };

  // Memory bar elements
  private memoryElements: MemoryElements = {
    heapBar: null,
    heapValue: null,
    gpuBar: null,
    gpuValue: null,
    opfsBar: null,
    opfsValue: null,
    heapStackedBar: null,
    gpuStackedBar: null,
    totalValue: null,
  };

  // Attention kernel UI
  private attentionKernelSelect: HTMLSelectElement | null = null;
  private attentionKernelNote: HTMLElement | null = null;
  private manifestAttentionKernelDefault: string | null = null;

  // Memory control UI
  private unloadModelBtn: HTMLButtonElement | null = null;
  private clearMemoryBtn: HTMLButtonElement | null = null;
  private swapIndicator: HTMLElement | null = null;
  private runtimeKernelHints: KernelHints | null = null;
  private runtimeAttentionKernel: string | null = null;

  // Sampling controls
  private temperatureInput: HTMLInputElement | null = null;
  private topPInput: HTMLInputElement | null = null;
  private topKInput: HTMLInputElement | null = null;

  // Converter UI
  private convertBtn: HTMLButtonElement | null = null;
  private convertStatus: HTMLElement | null = null;
  private convertProgress: HTMLElement | null = null;
  private convertMessage: HTMLElement | null = null;
  private isConverting = false;

  // Memory polling interval
  private _memoryPollInterval: ReturnType<typeof setInterval> | null = null;

  /**
   * Initialize the application
   */
  async init(): Promise<void> {
    console.log('[DopplerDemo] Initializing...');

    // Get DOM references
    this.statusDot = document.querySelector('.status-dot');
    this.statusText = document.querySelector('.status-text');
    this.capabilitiesList = document.querySelector('#capabilities-list');
    this.statsElements = {
      tps: document.querySelector('#stat-tps'),
      memory: document.querySelector('#stat-memory'),
      gpu: document.querySelector('#stat-gpu'),
      kv: document.querySelector('#stat-kv'),
    };

    // GPU info elements
    this.gpuElements = {
      device: document.querySelector('#gpu-device'),
      vram: document.querySelector('#gpu-vram'),
      vramLabel: document.querySelector('#gpu-vram-label'),
      ram: document.querySelector('#gpu-ram'),
      ramRow: document.querySelector('#gpu-ram-row'),
      features: document.querySelector('#gpu-features'),
      unifiedNote: document.querySelector('#gpu-unified-note'),
    };

    // Memory bar elements
    this.memoryElements = {
      heapBar: document.querySelector('#memory-bar-heap'),
      heapValue: document.querySelector('#memory-heap'),
      gpuBar: document.querySelector('#memory-bar-gpu'),
      gpuValue: document.querySelector('#memory-gpu'),
      opfsBar: document.querySelector('#memory-bar-opfs'),
      opfsValue: document.querySelector('#memory-opfs'),
      // Stacked total bar
      heapStackedBar: document.querySelector('#memory-bar-heap-stacked'),
      gpuStackedBar: document.querySelector('#memory-bar-gpu-stacked'),
      totalValue: document.querySelector('#memory-total'),
    };

    this.attentionKernelSelect = document.querySelector('#attention-kernel-select');
    this.attentionKernelNote = document.querySelector('#attention-kernel-note');
    this._readRuntimeOverridesFromURL();

    // Memory control elements
    this.unloadModelBtn = document.querySelector('#unload-model-btn');
    this.clearMemoryBtn = document.querySelector('#clear-memory-btn');
    this.swapIndicator = document.querySelector('#swap-indicator');

    this.temperatureInput = document.querySelector('#temperature-input');
    this.topPInput = document.querySelector('#top-p-input');
    this.topKInput = document.querySelector('#top-k-input');

    // Converter elements
    this.convertBtn = document.querySelector('#convert-btn');
    this.convertStatus = document.querySelector('#convert-status');
    this.convertProgress = document.querySelector('#convert-progress');
    this.convertMessage = document.querySelector('#convert-message');

    // Initialize UI components
    this._initComponents();

    // Check WebGPU support
    await this._detectCapabilities();

    // Load cached models list
    await this._loadCachedModels();

    // Set initial status
    if (this.capabilities.webgpu) {
      this._setStatus('ready', 'Ready');
      this.chatUI?.setInputEnabled(false); // Disabled until model loaded
    } else {
      this._setStatus('error', 'WebGPU not supported');
      this._showError(
        'WebGPU is not available in this browser. Please use Chrome 113+, Edge 113+, or Firefox Nightly with WebGPU enabled.'
      );
    }

    console.log('[DopplerDemo] Initialized');
  }

  /**
   * Initialize UI components
   */
  private _initComponents(): void {
    const container = document.querySelector('#app') as HTMLElement;

    // Model Selector
    this.modelSelector = new ModelSelector(container, {
      onSelect: (model, opts) => this.selectModel(model as RegisteredModel, opts),
      onDownload: (model, opts) => this.downloadModel(model as RegisteredModel, opts),
      onDelete: (model) => this.deleteModel(model as RegisteredModel),
      onQuickStart: (model) => {
        // Use the remote source ID (e.g., 'gemma-1b-instruct') for QUICKSTART_MODELS lookup
        const modelId = model.sources?.remote?.id || model.key;
        this.startQuickStart(modelId);
      },
    });

    // Chat UI
    this.chatUI = new ChatUI(container, {
      onSend: (message) => this.chat(message),
      onStop: () => this.stopGeneration(),
      onClear: () => this.clearConversation(),
    });

    // Progress UI
    this.progressUI = new ProgressUI(container);

    // Quick-Start UI
    this.quickStartUI = new QuickStartUI(container, {
      onDownloadComplete: (modelId) => this._onQuickStartComplete(modelId),
      onRunModel: (modelId) => this._runQuickStartModel(modelId),
      onCancel: () => console.log('[QuickStart] Cancelled by user'),
    });

    // Attention kernel override dropdown
    if (this.attentionKernelSelect) {
      if (this.runtimeAttentionKernel) {
        this.attentionKernelSelect.value = this.runtimeAttentionKernel;
      }
      this.attentionKernelSelect.addEventListener('change', () => {
        const value = this.attentionKernelSelect!.value;
        if (this.pipeline && typeof (this.pipeline as Pipeline & { setAttentionKernel?: (v: string) => void }).setAttentionKernel === 'function') {
          (this.pipeline as Pipeline & { setAttentionKernel: (v: string) => void }).setAttentionKernel(value);
        }
        this._updateAttentionKernelNote();
      });
    }

    // Sampling inputs (clamp and persist)
    const clampNumber = (input: HTMLInputElement, min: number, max: number): void => {
      const n = parseFloat(input.value);
      if (!Number.isFinite(n)) return;
      input.value = Math.min(max, Math.max(min, n)).toString();
    };
    if (this.temperatureInput) {
      this.temperatureInput.addEventListener('change', () =>
        clampNumber(this.temperatureInput!, 0.1, 2.0)
      );
    }
    if (this.topPInput) {
      this.topPInput.addEventListener('change', () => clampNumber(this.topPInput!, 0, 1));
    }
    if (this.topKInput) {
      this.topKInput.addEventListener('change', () => clampNumber(this.topKInput!, 0, 200));
    }

    // Convert button
    if (this.convertBtn) {
      if (isConversionSupported()) {
        this.convertBtn.addEventListener('click', () => this._handleConvert());
      } else {
        this.convertBtn.disabled = true;
        this.convertBtn.title = 'Model conversion requires File System Access API (Chrome/Edge)';
      }
    }

    // Memory control buttons
    if (this.unloadModelBtn) {
      this.unloadModelBtn.addEventListener('click', () => this._unloadCurrentModel());
    }
    if (this.clearMemoryBtn) {
      this.clearMemoryBtn.addEventListener('click', () => this._clearAllMemory());
    }
  }

  private _readRuntimeOverridesFromURL(): void {
    const params = new URLSearchParams(window.location.search);
    const hints: KernelHints = {};

    const kernelHintsRaw = params.get('kernelHints');
    if (kernelHintsRaw) {
      try {
        const parsed = JSON.parse(kernelHintsRaw);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          Object.assign(hints, parsed as KernelHints);
        }
      } catch (err) {
        console.warn('[DopplerDemo] Failed to parse kernelHints from URL:', (err as Error).message);
      }
    }

    const computePrecision = params.get('computePrecision');
    const q4kMatmul = params.get('q4kMatmul');
    const f16Matmul = params.get('f16Matmul');
    const attentionPrefill = params.get('attentionPrefill');
    const attentionDecode = params.get('attentionDecode');

    if (computePrecision) hints.computePrecision = computePrecision as KernelHints['computePrecision'];
    if (q4kMatmul) hints.q4kMatmul = q4kMatmul as KernelHints['q4kMatmul'];
    if (f16Matmul) hints.f16Matmul = f16Matmul as KernelHints['f16Matmul'];
    if (attentionPrefill) hints.attentionPrefill = attentionPrefill as KernelHints['attentionPrefill'];
    if (attentionDecode) hints.attentionDecode = attentionDecode as KernelHints['attentionDecode'];

    if (Object.keys(hints).length > 0) {
      this.runtimeKernelHints = hints;
      console.log('[DopplerDemo] Runtime kernel hints from URL:', hints);
    }

    const attentionKernel = params.get('attentionKernel');
    if (attentionKernel) {
      this.runtimeAttentionKernel = attentionKernel;
    }
  }

  /**
   * Detect browser capabilities
   */
  private async _detectCapabilities(): Promise<void> {
    console.log('[DopplerDemo] Detecting capabilities...');

    // WebGPU
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          this.capabilities.webgpu = true;

          // Check features
          this.capabilities.f16 = adapter.features.has('shader-f16');
          this.capabilities.subgroups = adapter.features.has('subgroups');

          // Get adapter info for logging
          const info: Partial<GPUAdapterInfo> = (adapter as GPUAdapter & { info?: GPUAdapterInfo; requestAdapterInfo?: () => Promise<GPUAdapterInfo> }).info || (await (adapter as GPUAdapter & { requestAdapterInfo?: () => Promise<GPUAdapterInfo> }).requestAdapterInfo?.()) || {};
          console.log(
            '[DopplerDemo] GPU:',
            info.vendor || 'unknown',
            info.architecture || info.device || 'unknown'
          );

          // Populate GPU info panel
          this._populateGPUInfo(adapter, info as GPUAdapterInfo);
        }
      } catch (e) {
        console.warn('[DopplerDemo] WebGPU init failed:', e);
      }
    }

    // Start memory stats polling
    this._startMemoryPolling();

    // Memory64 (basic check)
    try {
      const memory64Test = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x05, 0x04, 0x01, 0x04, 0x01, 0x00,
      ]);
      await WebAssembly.compile(memory64Test);
      this.capabilities.memory64 = true;
    } catch {
      this.capabilities.memory64 = false;
    }

    // Update UI
    this._updateCapabilitiesUI();
  }

  /**
   * Update capabilities list UI
   */
  private _updateCapabilitiesUI(): void {
    if (!this.capabilitiesList) return;
    const items = this.capabilitiesList.querySelectorAll('li');
    items.forEach((item) => {
      const cap = (item as HTMLElement).dataset.cap as keyof Capabilities;
      if (this.capabilities[cap]) {
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
   */
  private _resolveGPUName(info: GPUAdapterInfo): string {
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
      console.log(`[GPU] Unknown device: vendor=${vendor}, device=${device}, arch=${arch}`);
      // Capitalize vendor
      const vendorName = vendor.charAt(0).toUpperCase() + vendor.slice(1);
      return `${vendorName} GPU`;
    }

    return 'Unknown GPU';
  }

  /**
   * Populate GPU info panel with adapter details
   */
  private _populateGPUInfo(adapter: GPUAdapter, info: GPUAdapterInfo): void {
    if (!this.gpuElements.device) return;

    // Device name with friendly resolution
    const deviceName = this._resolveGPUName(info);
    this.gpuElements.device.textContent = deviceName;
    this.gpuElements.device.title = deviceName;

    // Detect unified memory architecture (Apple Silicon, etc)
    const isUnifiedMemory = this._isUnifiedMemoryArchitecture(info);

    // System RAM (for unified memory systems)
    // Note: navigator.deviceMemory caps at 8GB for privacy
    // Override with ?ram=24 URL param to show actual RAM
    const urlParams = new URLSearchParams(window.location.search);
    const ramOverride = urlParams.get('ram');
    const deviceMemoryGB = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
    if (isUnifiedMemory && this.gpuElements.ramRow && this.gpuElements.ram) {
      if (ramOverride) {
        this.gpuElements.ram.textContent = `${ramOverride} GB`;
      } else if (deviceMemoryGB && deviceMemoryGB >= 8) {
        // 8GB is the max reported, actual RAM is likely higher
        this.gpuElements.ram.textContent = '8+ GB';
      } else if (deviceMemoryGB) {
        this.gpuElements.ram.textContent = `${deviceMemoryGB} GB`;
      } else {
        this.gpuElements.ram.textContent = 'Unknown';
      }
      this.gpuElements.ramRow.hidden = false;
    }

    // Buffer limit (WebGPU's max buffer size - conservative on unified memory)
    const limits = (adapter.limits || {}) as GPUSupportedLimits & { maxBufferSize?: number; maxStorageBufferBindingSize?: number };
    const maxBufferSize = limits.maxBufferSize || 0;
    const maxStorageSize = limits.maxStorageBufferBindingSize || 0;
    const bufferLimit = Math.max(maxBufferSize, maxStorageSize);
    if (bufferLimit > 0) {
      this.gpuElements.vram!.textContent = this._formatBytes(bufferLimit);
    } else {
      this.gpuElements.vram!.textContent = 'Unknown';
    }

    // Show unified memory note
    if (isUnifiedMemory && this.gpuElements.unifiedNote) {
      this.gpuElements.unifiedNote.hidden = false;
    }

    // Features
    const features: string[] = [];
    if (this.capabilities.f16) features.push('F16');
    if (this.capabilities.subgroups) features.push('Subgroups');
    if (adapter.features.has('timestamp-query')) features.push('Timestamps');
    this.gpuElements.features!.textContent = features.length > 0 ? features.join(', ') : 'Basic';
  }

  /**
   * Detect unified memory architecture (Apple Silicon, integrated GPUs)
   */
  private _isUnifiedMemoryArchitecture(info: GPUAdapterInfo): boolean {
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
  private _startMemoryPolling(): void {
    // Update immediately
    this._updateMemoryStats();

    // Poll every 2 seconds
    this._memoryPollInterval = setInterval(() => {
      this._updateMemoryStats();
    }, 2000);
  }

  /**
   * Update memory stats display
   */
  private _updateMemoryStats(): void {
    if (!this.memoryElements.heapBar) return;

    let usedHeap = 0;
    let usedGpu = 0;
    let totalLimit = 0;

    // JS Heap (from performance.memory if available - Chrome only)
    const memory = (performance as Performance & { memory?: {
      usedJSHeapSize?: number;
      jsHeapSizeLimit?: number;
      totalJSHeapSize?: number;
    } }).memory;
    if (memory) {
      usedHeap = memory.usedJSHeapSize || 0;
      const totalHeap = memory.jsHeapSizeLimit || memory.totalJSHeapSize || 1;
      const heapPercent = Math.min(100, (usedHeap / totalHeap) * 100);
      totalLimit = totalHeap;

      this.memoryElements.heapBar.style.width = `${heapPercent}%`;
      this.memoryElements.heapValue!.textContent = this._formatBytes(usedHeap);
    } else {
      this.memoryElements.heapValue!.textContent = 'N/A';
    }

    // GPU buffer usage (from buffer pool)
    try {
      const bufferPool = getBufferPool();
      const poolStats = bufferPool.getStats();
      usedGpu = poolStats.currentBytesAllocated || 0;
      // Use peak as a rough limit, or device max buffer size
      const gpuLimit = Math.max(poolStats.peakBytesAllocated, 4 * 1024 * 1024 * 1024); // At least 4GB scale
      const gpuPercent = Math.min(100, (usedGpu / gpuLimit) * 100);

      this.memoryElements.gpuBar!.style.width = `${gpuPercent}%`;
      this.memoryElements.gpuValue!.textContent = this._formatBytes(usedGpu);

      // Use GPU limit for total if larger than heap limit
      if (gpuLimit > totalLimit) totalLimit = gpuLimit;
    } catch {
      this.memoryElements.gpuValue!.textContent = '--';
    }

    // OPFS cache storage (async, but we'll update on next cycle)
    if (this.memoryElements.opfsBar && this.memoryElements.opfsValue) {
      navigator.storage.estimate().then((estimate) => {
        const opfsUsed = estimate.usage || 0;
        const opfsQuota = estimate.quota || 1;
        const opfsPercent = Math.min(100, (opfsUsed / opfsQuota) * 100);

        this.memoryElements.opfsBar!.style.width = `${opfsPercent}%`;
        this.memoryElements.opfsValue!.textContent = this._formatBytes(opfsUsed);
      }).catch(() => {
        this.memoryElements.opfsValue!.textContent = '--';
      });
    }

    // Update stacked total bar
    if (this.memoryElements.heapStackedBar && this.memoryElements.gpuStackedBar) {
      const totalUsed = usedHeap + usedGpu;
      // Calculate percentages relative to combined limit (or reasonable max)
      const combinedLimit = Math.max(totalLimit, 8 * 1024 * 1024 * 1024); // At least 8GB scale
      const heapStackedPercent = Math.min(50, (usedHeap / combinedLimit) * 100);
      const gpuStackedPercent = Math.min(50, (usedGpu / combinedLimit) * 100);

      this.memoryElements.heapStackedBar.style.width = `${heapStackedPercent}%`;
      this.memoryElements.gpuStackedBar.style.width = `${gpuStackedPercent}%`;

      if (this.memoryElements.totalValue) {
        this.memoryElements.totalValue.textContent = this._formatBytes(totalUsed);
      }

      // Detect potential swapping (total > device memory)
      // navigator.deviceMemory gives RAM in GB (approximate)
      const deviceMemoryGB = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
      if (deviceMemoryGB && this.swapIndicator) {
        const physicalRamBytes = deviceMemoryGB * 1024 * 1024 * 1024;
        // Show swap warning if we're using more than 90% of reported device memory
        // (deviceMemory is capped/rounded, so actual RAM may be higher)
        if (totalUsed > physicalRamBytes * 0.9) {
          this.swapIndicator.hidden = false;
        } else {
          this.swapIndicator.hidden = true;
        }
      }
    }
  }

  /**
   * Format bytes to human-readable string
   */
  private _formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  /**
   * Format model ID to a human-readable name
   */
  private _formatModelName(modelId: string): string {
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
   */
  private _getModelKey(arch: string | undefined, quant: string | undefined, _size?: number | string): string {
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
   */
  private async _loadCachedModels(): Promise<void> {
    console.log('[DopplerDemo] Discovering models...');

    // Map to deduplicate models: key -> model info with sources
    const modelMap = new Map<string, RegisteredModel>();

    // Helper to add/merge a model into the map
    const addModel = (
      key: string,
      info: Partial<RegisteredModel>,
      sourceType: keyof ModelSources,
      sourceData: { id: string; url?: string }
    ): void => {
      if (modelMap.has(key)) {
        // Merge sources
        const existing = modelMap.get(key)!;
        (existing.sources as Record<string, unknown>)[sourceType] = sourceData;
        // Prefer better metadata (server > browser > remote)
        if (sourceType === 'server' || (sourceType === 'browser' && !existing.sources.server)) {
          existing.name = info.name || existing.name;
          existing.size = info.size || existing.size;
          existing.downloadSize = info.downloadSize || existing.downloadSize;
        }
      } else {
        modelMap.set(key, {
          ...info,
          key,
          sources: { [sourceType]: sourceData },
        } as RegisteredModel);
      }
    };

    // 1. Discover server models (local HTTP)
    const serverModels = await discoverLocalModels();
    console.log(`[DopplerDemo] Found ${serverModels.length} server models`);

    for (const model of serverModels) {
      const key = this._getModelKey(model.architecture, model.quantization, model.downloadSize);
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
    let cachedIds: string[] = [];
    try {
      cachedIds = await listModels();
      console.log('[DopplerDemo] Found cached models in OPFS:', cachedIds);
    } catch (err) {
      console.warn('[DopplerDemo] Could not query cached models:', (err as Error).message);
    }

    for (const cachedId of cachedIds) {
      try {
        await openModelDirectory(cachedId);
        const manifestText = await loadManifestFromOPFS();
        if (manifestText) {
          const manifest = parseManifest(manifestText);
          const config = manifest.config || {};
          const textConfig = (config as Record<string, unknown>).text_config || config;

          const arch = manifest.architecture || (config as Record<string, string[]>).architectures?.[0] || '';
          const quant = manifest.quantization || 'Unknown';
          const totalSize = (manifest.shards || []).reduce((sum, s) => sum + (s.size || 0), 0);

          // Estimate param count
          const hiddenSize = (textConfig as Record<string, number>).hidden_size || 0;
          let paramStr = 'Unknown';
          if (hiddenSize >= 4096) paramStr = '7B+';
          else if (hiddenSize >= 2048) paramStr = '1-3B';
          else if (hiddenSize >= 1024) paramStr = '<1B';

          const key = this._getModelKey(arch as string, quant, totalSize);
          addModel(
            key,
            {
              name: manifest.name || this._formatModelName(cachedId),
              architecture: arch as string,
              size: paramStr,
              quantization: quant,
              downloadSize: totalSize,
            },
            'browser',
            { id: cachedId }
          );
        }
      } catch (e) {
        console.warn(
          `[DopplerDemo] Could not load manifest for cached model ${cachedId}:`,
          (e as Error).message
        );
      }
    }

    // 3. Add remote models (available for download)
    for (const remote of REMOTE_MODELS) {
      const key = this._getModelKey(remote.architecture || remote.id, remote.quantization, remote.downloadSize);
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
      const key = this._getModelKey(req.architecture || modelId, req.quantization, req.downloadSize);
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
    const getAvailabilityScore = (m: RegisteredModel): number => {
      let score = 0;
      if (m.sources.server) score += 2;
      if (m.sources.browser) score += 1;
      return score;
    };

    MODEL_REGISTRY = Array.from(modelMap.values()).sort((a, b) => {
      return getAvailabilityScore(b) - getAvailabilityScore(a);
    });

    console.log(`[DopplerDemo] Model registry: ${MODEL_REGISTRY.length} unique models`);
    this.modelSelector?.setModels(MODEL_REGISTRY as ModelInfo[]);
  }

  /**
   * Select and load a model (run it)
   */
  async selectModel(
    modelOrKey: RegisteredModel | string,
    opts: { preferredSource?: string } = {}
  ): Promise<void> {
    if (this.isGenerating) {
      this._showError('Cannot switch models while generating');
      return;
    }

    // Support both model object and key string
    const model =
      typeof modelOrKey === 'string'
        ? MODEL_REGISTRY.find((m) => m.key === modelOrKey)
        : modelOrKey;

    if (!model) {
      this._showError(`Unknown model: ${modelOrKey}`);
      return;
    }

    const sources = model.sources || {};
    const hasServer = !!sources.server;
    const hasBrowser = !!sources.browser;

    if (!hasServer && !hasBrowser) {
      this._showError('Model not available locally. Download it first.');
      return;
    }

    // Use preferred source if specified, otherwise default to server > browser
    let useServer: boolean;
    if (opts.preferredSource === 'server' && hasServer) {
      useServer = true;
    } else if (opts.preferredSource === 'browser' && hasBrowser) {
      useServer = false;
    } else {
      useServer = hasServer; // Default: prefer server
    }

    const sourceInfo = useServer ? sources.server! : sources.browser!;
    const sourceType = useServer ? 'server' : 'browser';

    console.log(`[DopplerDemo] Loading model: ${model.name} from ${sourceType}`);
    this._setStatus('loading', 'Loading model...');
    this.progressUI?.show('Loading model...');

    try {
      // Unload current model if any
      if (this.pipeline) {
        if (typeof (this.pipeline as Pipeline & { unload?: () => Promise<void> }).unload === 'function') {
          await (this.pipeline as Pipeline & { unload: () => Promise<void> }).unload();
        }
        this.pipeline = null;
      }

      let manifest: RDRRManifest;
      let loadShardFn: (idx: number) => Promise<ArrayBuffer>;

      // Track loading source for progress - distinguish between network/disk/cache
      const isLocalServer = sourceInfo.url?.match(/^(https?:\/\/)?(localhost|127\.0\.0\.1|0\.0\.0\.0|file:)/i);
      const sourceType = useServer ? (isLocalServer ? 'disk' : 'network') : 'cache';
      this.progressUI?.setSourceType(sourceType as import('./progress-ui.js').SourceType);

      if (useServer) {
        // Load from HTTP - show source phase
        this.progressUI?.setPhaseProgress({ phase: 'source', percent: 5, message: 'Fetching manifest...' });
        const manifestUrl = `${sourceInfo.url}/manifest.json`;
        const response = await fetch(manifestUrl);
        if (!response.ok) throw new Error(`Failed to fetch manifest: ${response.status}`);
        manifest = parseManifest(await response.text());

        // Create HTTP shard loader
        loadShardFn = async (idx: number): Promise<ArrayBuffer> => {
          const shard = manifest.shards[idx];
          const shardUrl = `${sourceInfo.url}/${shard.filename}`;
          const res = await fetch(shardUrl);
          if (!res.ok) throw new Error(`Failed to fetch shard ${idx}: ${res.status}`);
          return await res.arrayBuffer();
        };
      } else {
        // Load from OPFS (browser cache) - show source phase (labeled "Cache")
        await openModelDirectory(sourceInfo.id);
        this.progressUI?.setPhaseProgress({ phase: 'source', percent: 5, message: 'Loading manifest...' });
        const manifestJson = await loadManifestFromOPFS();
        manifest = parseManifest(manifestJson);

        // Create OPFS shard loader
        const { loadShard } = await import('../storage/shard-manager.js');
        loadShardFn = (idx: number) => loadShard(idx);
      }

      // Initialize GPU - show GPU phase starting
      this.progressUI?.setPhaseProgress({ phase: 'gpu', percent: 5, message: 'Initializing...' });

      // Capture manifest default attention kernel preference.
      this.manifestAttentionKernelDefault =
        (manifest as RDRRManifest & { optimizations?: { attentionKernel?: string }; attentionKernel?: string; runtime?: { attentionKernel?: string } }).optimizations?.attentionKernel ||
        (manifest as RDRRManifest & { attentionKernel?: string }).attentionKernel ||
        (manifest as RDRRManifest & { runtime?: { attentionKernel?: string } }).runtime?.attentionKernel ||
        null;

      // Ensure GPU device is initialized
      const device = getDevice() || (await initDevice());
      const gpuCaps = getKernelCapabilities();
      const memCaps = await getMemoryCapabilities();
      const heapManager = getHeapManager();
      await heapManager.init();

      this.progressUI?.setPhaseProgress({ phase: 'gpu', percent: 10, message: 'Creating pipeline...' });

      // Create pipeline with multi-phase progress tracking
      this.pipeline = await createPipeline(manifest as unknown as import('../inference/pipeline/config.js').Manifest, {
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
          attentionKernel: this.runtimeAttentionKernel || this.attentionKernelSelect?.value || 'auto',
          debug: new URLSearchParams(window.location.search).has('debug'),
          kernelHints: this.runtimeKernelHints || undefined,
        },
        onProgress: (progress: {
          percent: number;
          message?: string;
          stage?: string;
          layer?: number;
          total?: number;
          bytesLoaded?: number;
          totalBytes?: number;
          bytesPerSecond?: number;
        }) => {
          const stage = progress.stage || 'layers';

          // Map loader stages to UI phases (simplified: source + gpu)
          if (stage === 'manifest' || stage === 'shards') {
            // Shard loading from source
            this.progressUI?.setPhaseProgress({
              phase: 'source',
              percent: Math.min(100, progress.percent * 1.2), // Scale to show some progress
              bytesLoaded: progress.bytesLoaded,
              totalBytes: progress.totalBytes,
              speed: progress.bytesPerSecond,
            });
          } else if (stage === 'layers' || stage === 'gpu_transfer') {
            // Mark source complete, show GPU progress
            this.progressUI?.setPhaseProgress({ phase: 'source', percent: 100, message: 'Complete' });

            // GPU phase: show layer progress
            const gpuPercent = 10 + (progress.percent * 0.9);
            let message: string;
            if (progress.layer !== undefined && progress.total) {
              message = `Layer ${progress.layer}/${progress.total}`;
            } else if (stage === 'gpu_transfer') {
              message = 'Uploading weights...';
            } else {
              message = `${Math.round(gpuPercent)}%`;
            }
            this.progressUI?.setPhaseProgress({
              phase: 'gpu',
              percent: gpuPercent,
              message,
            });
          } else if (stage === 'complete') {
            // All phases complete
            this.progressUI?.setPhaseProgress({ phase: 'source', percent: 100, message: 'Done' });
            this.progressUI?.setPhaseProgress({ phase: 'gpu', percent: 100, message: 'Ready' });
          }
        },
      });

      this.currentModel = model;
      this.modelSelector?.setActiveModel(model.key);
      this.progressUI?.hide();
      this._setStatus('ready', `${model.name} loaded`);
      this.chatUI?.setInputEnabled(true);
      this.chatUI?.focusInput();
      this._updateAttentionKernelNote();
      this._updateInitialStats();

      console.log(`[DopplerDemo] Model loaded: ${model.name} (${model.key})`);
    } catch (error) {
      console.error('[DopplerDemo] Model load failed:', error);
      this.progressUI?.hide();
      this._setStatus('error', 'Load failed');
      this._showError(`Failed to load model: ${(error as Error).message}`);
    }
  }

  /**
   * Download/cache a model to browser storage
   */
  async downloadModel(
    model: RegisteredModel,
    opts: { runAfter?: boolean } = {}
  ): Promise<void> {
    const sources = model.sources || {};

    // Determine URL: prefer server (for caching), then remote
    let downloadUrl: string | null = null;
    let storageId = model.key.replace(/[^a-zA-Z0-9_-]/g, '_'); // Safe filename

    if (sources.server) {
      downloadUrl = sources.server.url!;
    } else if (sources.remote) {
      downloadUrl = sources.remote.url!;
      storageId = sources.remote.id || storageId;
    }

    if (!downloadUrl) {
      this._showError('No download source available');
      return;
    }

    console.log(`[DopplerDemo] Downloading "${model.name}" from: ${downloadUrl}`);
    this._setStatus('loading', `Downloading ${model.name}...`);

    try {
      const success = await downloadModel(
        downloadUrl,
        (progress: DownloadProgress) => {
          const percent =
            progress.totalBytes > 0
              ? Math.round((progress.downloadedBytes / progress.totalBytes) * 100)
              : 0;
          this.modelSelector?.setDownloadProgress(model.key, percent);

          if (progress.stage === 'verifying') {
            this._setStatus('loading', 'Verifying...');
          }
        },
        { modelId: storageId }
      );

      if (!success) {
        throw new Error('Download failed');
      }

      this._setStatus('ready', 'Download complete');

      // Refresh models list to update sources
      await this._loadCachedModels();

      console.log(`[DopplerDemo] Download complete: ${model.name}`);

      // Run after download if requested
      if (opts.runAfter) {
        // Find the updated model in registry
        const updatedModel = MODEL_REGISTRY.find((m) => m.key === model.key);
        if (updatedModel) {
          await this.selectModel(updatedModel);
        }
      }
    } catch (error) {
      console.error('[DopplerDemo] Download failed:', error);
      this.modelSelector?.setDownloadProgress(model.key, 0);
      this._setStatus('error', 'Download failed');
      this._showError(`Download failed: ${(error as Error).message}`);
    }
  }

  /**
   * Delete a model from browser cache
   */
  async deleteModel(model: RegisteredModel): Promise<void> {
    const sources = model.sources || {};
    const browserId = sources.browser?.id;

    if (!browserId) {
      this._showError('Model is not cached in browser');
      return;
    }

    console.log(`[DopplerDemo] Deleting cached model: ${model.name} (${browserId})`);

    try {
      // Unload if currently active
      if (this.currentModel?.key === model.key) {
        if (this.pipeline) {
          if (typeof (this.pipeline as Pipeline & { unload?: () => Promise<void> }).unload === 'function') {
            await (this.pipeline as Pipeline & { unload: () => Promise<void> }).unload();
          }
          this.pipeline = null;
        }
        this.currentModel = null;
        this.modelSelector?.setActiveModel(null);
        this.chatUI?.setInputEnabled(false);
      }

      // Delete from OPFS
      await deleteModelFromOPFS(browserId);
      this._setStatus('ready', 'Cache cleared');

      // Refresh models list
      await this._loadCachedModels();
    } catch (error) {
      console.error('[DopplerDemo] Delete failed:', error);
      this._showError(`Delete failed: ${(error as Error).message}`);
    }
  }

  /**
   * Send a chat message and generate response
   */
  async chat(message: string): Promise<void> {
    if (!this.currentModel) {
      this._showError('No model loaded');
      return;
    }

    if (!this.pipeline) {
      this._showError('Pipeline not initialized');
      return;
    }

    if (this.isGenerating) {
      return;
    }

    console.log(`[DopplerDemo] Generating response...`);
    this.isGenerating = true;
    this.abortController = new AbortController();

    // Add user message
    this.chatUI?.addMessage('user', message);

    // Start streaming response
    this.chatUI?.startStream();
    this._setStatus('loading', 'Generating...');

    try {
      // Use real pipeline generation
      let tokenCount = 0;
      const startTime = performance.now();

      for await (const token of this.pipeline.generate(message, {
        maxTokens: 512,
        temperature: this._getSamplingTemperature(),
        topP: this._getSamplingTopP(),
        topK: this._getSamplingTopK(),
        signal: this.abortController.signal,
      })) {
        if (this.abortController.signal.aborted) break;
        this.chatUI?.streamToken(token);
        tokenCount++;

        // Update TPS periodically
        if (tokenCount % 10 === 0) {
          const elapsed = (performance.now() - startTime) / 1000;
          this._updateStats(tokenCount / elapsed);
        }
      }

      const stats = this.chatUI?.finishStream();
      if (stats) {
        this._updateStats(stats.tokensPerSec);
      }
      this._setStatus('ready', `${this.currentModel.name}`);
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        this.chatUI?.cancelStream();
        this._setStatus('ready', 'Stopped');
      } else {
        console.error('[DopplerDemo] Generation error:', error);
        this.chatUI?.cancelStream();
        this._setStatus('error', 'Generation failed');
        this._showError(`Generation failed: ${(error as Error).message}`);
      }
    } finally {
      this.isGenerating = false;
      this.abortController = null;
    }
  }

  private _getSamplingTemperature(): number {
    const n = parseFloat(this.temperatureInput?.value || '');
    return Number.isFinite(n) ? n : 0.7;
  }

  private _getSamplingTopP(): number {
    const n = parseFloat(this.topPInput?.value || '');
    return Number.isFinite(n) ? n : 0.9;
  }

  private _getSamplingTopK(): number {
    const n = parseInt(this.topKInput?.value || '', 10);
    return Number.isFinite(n) ? n : 40;
  }

  /**
   * Stop current generation
   */
  stopGeneration(): void {
    if (this.abortController) {
      this.abortController.abort();
    }
  }

  /**
   * Clear conversation history
   */
  clearConversation(): void {
    if (this.pipeline && typeof (this.pipeline as Pipeline & { clearKVCache?: () => void }).clearKVCache === 'function') {
      (this.pipeline as Pipeline & { clearKVCache: () => void }).clearKVCache();
    }
    this.chatUI?.clear();
    console.log('[DopplerDemo] Conversation cleared');
  }

  /**
   * Get current status
   */
  getStatus(): {
    model: string | null;
    modelName: string | null;
    isGenerating: boolean;
    capabilities: Capabilities;
    memory: unknown;
    gpu: unknown;
  } {
    let memoryUsage: unknown = null;
    let gpuUsage: unknown = null;

    if (this.pipeline) {
      // Get memory stats from pipeline if available
      if (typeof (this.pipeline as Pipeline & { getMemoryStats?: () => unknown }).getMemoryStats === 'function') {
        memoryUsage = (this.pipeline as Pipeline & { getMemoryStats: () => unknown }).getMemoryStats();
      }
      // Get GPU stats if available
      if (typeof (this.pipeline as Pipeline & { getGPUStats?: () => unknown }).getGPUStats === 'function') {
        gpuUsage = (this.pipeline as Pipeline & { getGPUStats: () => unknown }).getGPUStats();
      }
    }

    return {
      model: this.currentModel?.key || null,
      modelName: this.currentModel?.name || null,
      isGenerating: this.isGenerating,
      capabilities: { ...this.capabilities },
      memory: memoryUsage,
      gpu: gpuUsage,
    };
  }

  /**
   * Set status indicator
   */
  private _setStatus(state: string, text: string): void {
    if (this.statusDot) {
      this.statusDot.className = `status-dot ${state}`;
    }
    if (this.statusText) {
      this.statusText.textContent = text;
    }
  }

  /**
   * Update performance stats
   */
  private _updateStats(tps: number): void {
    if (this.statsElements.tps) {
      this.statsElements.tps.textContent = tps.toFixed(1);
    }

    // Update memory and GPU stats from pipeline
    if (this.pipeline) {
      if (this.statsElements.memory && typeof (this.pipeline as Pipeline & { getMemoryStats?: () => { used?: number } }).getMemoryStats === 'function') {
        const memStats = (this.pipeline as Pipeline & { getMemoryStats: () => { used?: number } }).getMemoryStats();
        if (memStats && memStats.used) {
          const usedMB = (memStats.used / 1024 / 1024).toFixed(0);
          this.statsElements.memory.textContent = `${usedMB} MB`;
        }
      }

      if (this.statsElements.kv && typeof (this.pipeline as Pipeline & { getKVCacheStats?: () => { seqLen: number; maxSeqLen: number } }).getKVCacheStats === 'function') {
        const kvStats = (this.pipeline as Pipeline & { getKVCacheStats: () => { seqLen: number; maxSeqLen: number } }).getKVCacheStats();
        if (kvStats) {
          this.statsElements.kv.textContent = `${kvStats.seqLen}/${kvStats.maxSeqLen}`;
        }
      }
    }
  }

  /**
   * Update initial stats after model load (before generation starts)
   */
  private _updateInitialStats(): void {
    // Show TPS as "Ready" before generation starts
    if (this.statsElements.tps) {
      this.statsElements.tps.textContent = 'Ready';
    }

    // Show memory usage
    if (this.statsElements.memory) {
      const heap = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize;
      if (heap) {
        const usedMB = (heap / 1024 / 1024).toFixed(0);
        this.statsElements.memory.textContent = `${usedMB} MB`;
      }
    }

    // Show GPU buffer count if available
    if (this.statsElements.gpu && this.pipeline) {
      const bufferPool = (this.pipeline as Pipeline & { getBufferPool?: () => { getStats?: () => { activeBuffers: number } } }).getBufferPool?.();
      if (bufferPool && typeof bufferPool.getStats === 'function') {
        const stats = bufferPool.getStats();
        this.statsElements.gpu.textContent = `${stats.activeBuffers}`;
      } else {
        this.statsElements.gpu.textContent = 'N/A';
      }
    }

    // Show KV cache as "0/max" initially
    if (this.statsElements.kv && this.pipeline) {
      const kvStats = (this.pipeline as Pipeline & { getKVCacheStats?: () => { seqLen: number; maxSeqLen: number } }).getKVCacheStats?.();
      if (kvStats) {
        this.statsElements.kv.textContent = `${kvStats.seqLen}/${kvStats.maxSeqLen}`;
      } else {
        this.statsElements.kv.textContent = '0';
      }
    }

    // Enable unload button
    if (this.unloadModelBtn) {
      this.unloadModelBtn.disabled = false;
    }
  }

  /**
   * Unload the current model from GPU memory
   */
  private async _unloadCurrentModel(): Promise<void> {
    if (!this.pipeline) {
      console.log('[DopplerDemo] No model loaded');
      return;
    }

    console.log('[DopplerDemo] Unloading current model...');
    this._setStatus('loading', 'Unloading model...');

    try {
      // Call pipeline unload if available
      if (typeof (this.pipeline as Pipeline & { unload?: () => Promise<void> }).unload === 'function') {
        await (this.pipeline as Pipeline & { unload: () => Promise<void> }).unload();
      }

      this.pipeline = null;
      this.currentModel = null;
      this.modelSelector?.setActiveModel(null);
      this.chatUI?.setInputEnabled(false);

      // Reset stats
      if (this.statsElements.tps) this.statsElements.tps.textContent = '--';
      if (this.statsElements.memory) this.statsElements.memory.textContent = '--';
      if (this.statsElements.gpu) this.statsElements.gpu.textContent = '--';
      if (this.statsElements.kv) this.statsElements.kv.textContent = '--';

      // Disable unload button
      if (this.unloadModelBtn) {
        this.unloadModelBtn.disabled = true;
      }

      this._setStatus('ready', 'Model unloaded');
      this._updateMemoryStats();
      console.log('[DopplerDemo] Model unloaded');
    } catch (error) {
      console.error('[DopplerDemo] Unload failed:', error);
      this._setStatus('error', 'Unload failed');
    }
  }

  /**
   * Clear all GPU memory (buffers, caches, heap)
   */
  private async _clearAllMemory(): Promise<void> {
    console.log('[DopplerDemo] Clearing all memory...');
    this._setStatus('loading', 'Clearing memory...');

    try {
      // First unload the model if loaded
      if (this.pipeline) {
        await this._unloadCurrentModel();
      }

      // Destroy buffer pool
      const { destroyBufferPool } = await import('../gpu/buffer-pool.js');
      destroyBufferPool();

      // Reset heap manager
      const { getHeapManager } = await import('../memory/heap-manager.js');
      const heapManager = getHeapManager();
      if (heapManager && typeof heapManager.reset === 'function') {
        heapManager.reset();
      }

      // Force garbage collection hint (may or may not work)
      if (typeof (globalThis as typeof globalThis & { gc?: () => void }).gc === 'function') {
        (globalThis as typeof globalThis & { gc: () => void }).gc();
      }

      this._setStatus('ready', 'Memory cleared');
      this._updateMemoryStats();
      console.log('[DopplerDemo] All memory cleared');
    } catch (error) {
      console.error('[DopplerDemo] Clear memory failed:', error);
      this._setStatus('error', 'Clear failed');
    }
  }

  /**
   * Update attention kernel note based on dropdown and manifest default.
   */
  private _updateAttentionKernelNote(): void {
    if (!this.attentionKernelNote || !this.attentionKernelSelect) return;

    const selected = this.attentionKernelSelect.value;
    const manifestDefault = this.manifestAttentionKernelDefault;

    if (selected && selected !== 'auto') {
      this.attentionKernelNote.textContent = `Override: ${selected}`;
    } else if (manifestDefault) {
      this.attentionKernelNote.textContent = `Manifest default: ${manifestDefault} (auto enabled)`;
    } else {
      this.attentionKernelNote.textContent = 'Auto selection enabled';
    }
  }

  /**
   * Show error modal
   */
  private _showError(message: string): void {
    const modal = document.querySelector('#error-modal') as HTMLElement | null;
    const messageEl = document.querySelector('#error-message') as HTMLElement | null;
    const closeBtn = document.querySelector('#error-close') as HTMLElement | null;

    if (messageEl) {
      messageEl.textContent = message;
    }
    if (modal) {
      modal.hidden = false;
    }

    const close = (): void => {
      if (modal) {
        modal.hidden = true;
      }
      closeBtn?.removeEventListener('click', close);
    };
    closeBtn?.addEventListener('click', close);
  }

  /**
   * Handle model conversion
   */
  private async _handleConvert(): Promise<void> {
    if (this.isConverting) {
      return;
    }

    try {
      // Pick files
      const files = await pickModelFiles();
      if (!files || files.length === 0) {
        return;
      }

      console.log(`[DopplerDemo] Converting ${files.length} files...`);
      this.isConverting = true;
      if (this.convertBtn) {
        this.convertBtn.disabled = true;
      }

      // Show progress UI
      if (this.convertStatus) {
        this.convertStatus.hidden = false;
      }
      this._updateConvertProgress(0, 'Starting conversion...');

      // Convert model
      const modelId = await convertModel(files, {
        onProgress: (progress: ConvertProgress) => {
          const percent = progress.percent || 0;
          const message = progress.message || progress.stage;
          this._updateConvertProgress(percent, message);

          if (progress.stage === ConvertStage.ERROR) {
            throw new Error(progress.message);
          }
        },
      });

      console.log(`[DopplerDemo] Conversion complete: ${modelId}`);
      this._updateConvertProgress(100, `Done! Model: ${modelId}`);

      // Refresh model list
      await this._loadCachedModels();

      // Hide progress after delay
      setTimeout(() => {
        if (this.convertStatus) {
          this.convertStatus.hidden = true;
        }
        this._updateConvertProgress(0, 'Ready');
      }, 3000);
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        console.log('[DopplerDemo] Conversion cancelled');
        this._updateConvertProgress(0, 'Cancelled');
      } else {
        console.error('[DopplerDemo] Conversion failed:', error);
        this._updateConvertProgress(0, `Error: ${(error as Error).message}`);
        this._showError(`Conversion failed: ${(error as Error).message}`);
      }
    } finally {
      this.isConverting = false;
      if (this.convertBtn) {
        this.convertBtn.disabled = false;
      }
    }
  }

  /**
   * Update conversion progress UI
   */
  private _updateConvertProgress(percent: number, message: string): void {
    if (this.convertProgress) {
      this.convertProgress.style.width = `${percent}%`;
    }
    if (this.convertMessage) {
      this.convertMessage.textContent = message;
    }
  }

  // ============================================================================
  // Quick-Start Methods
  // ============================================================================

  /**
   * Start quick-start flow for a model
   */
  async startQuickStart(modelId: string): Promise<void> {
    const config = QUICKSTART_MODELS[modelId];
    if (!config) {
      this._showError(`Unknown quick-start model: ${modelId}`);
      return;
    }

    console.log(`[QuickStart] Starting download for ${modelId}`);

    const result = await downloadQuickStartModel(modelId, {
      onPreflightComplete: (preflight) => {
        console.log('[QuickStart] Preflight:', preflight);

        // Show VRAM blocker if needed
        if (!preflight.vram.sufficient) {
          this.quickStartUI?.showVRAMBlocker(
            preflight.vram.required,
            preflight.vram.available
          );
        }
      },
      onStorageConsent: async (required, available, modelName) => {
        // Show consent dialog and wait for user response
        const consent = await this.quickStartUI?.showStorageConsent(
          modelName,
          required,
          available
        );
        if (consent) {
          this.quickStartUI?.showDownloadProgress();
        }
        return consent ?? false;
      },
      onProgress: (progress) => {
        this.quickStartUI?.setDownloadProgress(
          progress.percent,
          progress.downloadedBytes,
          progress.totalBytes,
          progress.speed
        );
      },
    });

    if (result.success) {
      this.quickStartUI?.showReady(modelId);
    } else if (result.blockedByPreflight) {
      // Already showing VRAM blocker
      console.log('[QuickStart] Blocked by preflight:', result.error);
    } else if (result.userDeclined) {
      console.log('[QuickStart] User declined');
      this.quickStartUI?.hide();
    } else {
      this.quickStartUI?.showError(result.error || 'Download failed');
    }
  }

  /**
   * Handle quick-start download completion
   */
  private async _onQuickStartComplete(modelId: string): Promise<void> {
    console.log(`[QuickStart] Download complete for ${modelId}`);
    // Refresh model list to show the downloaded model
    await this._loadCachedModels();
  }

  /**
   * Run model after quick-start download
   */
  private async _runQuickStartModel(modelId: string): Promise<void> {
    console.log(`[QuickStart] Running model ${modelId}`);

    // Find the model in registry and select it
    const model = MODEL_REGISTRY.find((m) => m.key === modelId || m.sources.browser?.id === modelId);
    if (model) {
      await this.selectModel(model);
    } else {
      // Refresh and try again
      await this._loadCachedModels();
      const refreshedModel = MODEL_REGISTRY.find((m) => m.key === modelId || m.sources.browser?.id === modelId);
      if (refreshedModel) {
        await this.selectModel(refreshedModel);
      } else {
        this._showError(`Model ${modelId} not found after download`);
      }
    }
  }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new DopplerDemo();
  app.init().catch(console.error);

  // Expose for debugging
  (window as Window & { dopplerDemo?: DopplerDemo }).dopplerDemo = app;
});

export default DopplerDemo;
