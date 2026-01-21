

import { log } from '../src/debug/index.js';
import { createConverterConfig, getRuntimeConfig, listPresets, MB } from '../src/config/index.js';

// UI Components
import { ModelSelector } from './model-selector.js';
import { ChatUI } from './chat-ui.js';
import { ProgressUI } from './progress-ui.js';
import { QuickStartUI } from './quickstart-ui.js';

// Controllers
import { CapabilitiesDetector } from './capabilities-detector.js';
import { MemoryMonitor } from './memory-monitor.js';
import { ModelRegistry } from './model-registry.js';
import { ModelLoader } from './model-loader.js';
import { ModelDownloader } from './model-downloader.js';
import { ChatController } from './chat-controller.js';
import { ConverterController } from './converter-controller.js';
import { DiagnosticsController } from './diagnostics-controller.js';
import { QuickStartController } from './quickstart-controller.js';
import { WorkspaceController } from './workspace-controller.js';
import { ToolingController } from './tooling-controller.js';

// UI Helpers
import {
  setStatus,
  updateCapabilitiesUI,
  populateGPUInfo,
  updateStats,
  resetStats,
  showError,
  clampInputValue,
} from './ui-helpers.js';

// ============================================================================
// Main Demo Application
// ============================================================================

export class DopplerDemo {
  // Controllers
  #capabilitiesDetector = null;
  #memoryMonitor = null;
  #modelRegistry = null;
  #modelLoader = null;
  #modelDownloader = null;
  #chatController = null;
  #converterController = null;
  #diagnosticsController = null;
  #quickStartController = null;
  #workspaceController = null;
  #toolingController = null;

  // UI Components
  #modelSelector = null;
  #chatUI = null;
  #progressUI = null;
  #quickStartUI = null;

  // Workspace UI
  #workspaceImportBtn = null;
  #workspaceRefreshBtn = null;
  #workspaceStatus = null;
  #workspaceFiles = null;
  #toolSelect = null;
  #toolArgs = null;
  #toolRunBtn = null;
  #toolRefreshBtn = null;
  #toolOutput = null;
  #toolStatus = null;

  // DOM references
  #statusDot = null;
  #statusText = null;
  #capabilitiesList = null;
  #statsElements = { tps: null, memory: null, gpu: null, kv: null };
  #gpuElements = {
    device: null, vram: null, vramLabel: null, ram: null,
    ramRow: null, features: null, unifiedNote: null,
  };
  #memoryElements = {
    heapBar: null, heapValue: null, gpuBar: null, gpuValue: null,
    kvBar: null, kvValue: null, opfsBar: null, opfsValue: null,
    headroomBar: null, headroomValue: null, heapStackedBar: null,
    gpuStackedBar: null, totalValue: null, swapIndicator: null,
  };

  // Memory control UI
  #unloadModelBtn = null;
  #clearMemoryBtn = null;

  // Sampling controls
  #temperatureInput = null;
  #topPInput = null;
  #topKInput = null;

  // Converter UI
  #convertBtn = null;
  #convertStatus = null;
  #convertProgress = null;
  #convertMessage = null;
  #convertUrlInput = null;
  #convertUrlBtn = null;
  #convertPresetSelect = null;
  #convertAllowDownloadInput = null;
  #convertMaxDownloadInput = null;

  // Diagnostics UI
  #diagnosticsModelSelect = null;
  #runtimePresetSelect = null;
  #runtimeConfigFileInput = null;
  #runtimeConfigClearBtn = null;
  #runtimeConfigStatus = null;
  #diagnosticsSuiteSelect = null;
  #diagnosticsRunBtn = null;
  #diagnosticsVerifyBtn = null;
  #diagnosticsExportBtn = null;
  #diagnosticsStatus = null;
  #diagnosticsReport = null;

  #diagnosticsRuntimeConfig = null;
  #diagnosticsRuntimeLabel = null;

  async init() {
    log.info('App', 'Initializing...');

    this.#initDOMReferences();
    this.#initControllers();
    this.#initUIComponents();
    this.#populateConverterPresets();
    this.#syncConverterUIFromConfig();
    this.#updateRuntimeConfigStatus();
    this.#initEventListeners();

    // Detect capabilities
    await this.#capabilitiesDetector.detect();
    const caps = this.#capabilitiesDetector.getState();

    // Update capabilities UI
    updateCapabilitiesUI(this.#capabilitiesList, caps);

    // Populate GPU info
    const adapter = this.#capabilitiesDetector.getAdapter();
    const adapterInfo = this.#capabilitiesDetector.getAdapterInfo();
    if (adapter && adapterInfo) {
      const limits = this.#capabilitiesDetector.getGPULimits();
      const isUnified = this.#capabilitiesDetector.isUnifiedMemoryArchitecture(adapterInfo);

      populateGPUInfo(this.#gpuElements, {
        deviceName: this.#capabilitiesDetector.resolveGPUName(adapterInfo),
        bufferLimitBytes: Math.max(limits?.maxBufferSize || 0, limits?.maxStorageSize || 0),
        isUnifiedMemory: isUnified,
        systemMemoryGB: navigator.deviceMemory,
        hasF16: caps.f16,
        hasSubgroups: caps.subgroups,
        hasTimestamps: this.#capabilitiesDetector.hasTimestampQuery(),
      });

      // Configure memory monitor
      const bufferLimit = Math.max(limits?.maxBufferSize || 0, limits?.maxStorageSize || 0);
      this.#memoryMonitor.configure({
        gpuBufferLimitBytes: bufferLimit > 0 ? bufferLimit : null,
        isUnifiedMemory: isUnified,
        estimatedSystemMemoryBytes: navigator.deviceMemory
          ? navigator.deviceMemory * 1024 * 1024 * 1024
          : null,
        getPipelineMemoryStats: () => this.#modelLoader.getMemoryStats(),
      });
    }

    // Start memory monitoring
    this.#memoryMonitor.start();

    // Discover models
    await this.#modelRegistry.discover();
    this.#modelSelector?.setModels(this.#modelRegistry.getModels());
    this.#updateDiagnosticsModels();

    try {
      await this.#workspaceController?.init();
    } catch (error) {
      log.warn('App', 'Workspace init failed', error);
    }

    // Set initial status
    if (caps.webgpu) {
      this.#setStatus('ready', 'Ready');
      this.#chatUI?.setInputEnabled(false);
    } else {
      this.#setStatus('error', 'WebGPU not supported');
      this.#showError(
        'WebGPU is not available in this browser. Please use Chrome 113+, Edge 113+, or Firefox Nightly with WebGPU enabled.'
      );
    }

    log.info('App', 'Initialized');
  }

  #initDOMReferences() {
    this.#statusDot = document.querySelector('.status-dot');
    this.#statusText = document.querySelector('.status-text');
    this.#capabilitiesList = document.querySelector('#capabilities-list');

    this.#statsElements = {
      tps: document.querySelector('#stat-tps'),
      memory: document.querySelector('#stat-memory'),
      gpu: document.querySelector('#stat-gpu'),
      kv: document.querySelector('#stat-kv'),
    };

    this.#gpuElements = {
      device: document.querySelector('#gpu-device'),
      vram: document.querySelector('#gpu-vram'),
      vramLabel: document.querySelector('#gpu-vram-label'),
      ram: document.querySelector('#gpu-ram'),
      ramRow: document.querySelector('#gpu-ram-row'),
      features: document.querySelector('#gpu-features'),
      unifiedNote: document.querySelector('#gpu-unified-note'),
    };

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
      heapStackedBar: document.querySelector('#memory-bar-heap-stacked'),
      gpuStackedBar: document.querySelector('#memory-bar-gpu-stacked'),
      totalValue: document.querySelector('#memory-total'),
      swapIndicator: document.querySelector('#swap-indicator'),
    };

    this.#unloadModelBtn = document.querySelector('#unload-model-btn');
    this.#clearMemoryBtn = document.querySelector('#clear-memory-btn');

    this.#temperatureInput = document.querySelector('#temperature-input');
    this.#topPInput = document.querySelector('#top-p-input');
    this.#topKInput = document.querySelector('#top-k-input');

    this.#convertBtn = document.querySelector('#convert-btn');
    this.#convertStatus = document.querySelector('#convert-status');
    this.#convertProgress = document.querySelector('#convert-progress');
    this.#convertMessage = document.querySelector('#convert-message');
    this.#convertUrlInput = document.querySelector('#convert-url-input');
    this.#convertUrlBtn = document.querySelector('#convert-url-btn');
    this.#convertPresetSelect = document.querySelector('#convert-model-preset');
    this.#convertAllowDownloadInput = document.querySelector('#convert-allow-download');
    this.#convertMaxDownloadInput = document.querySelector('#convert-max-download');

    this.#diagnosticsModelSelect = document.querySelector('#diagnostics-model');
    this.#runtimePresetSelect = document.querySelector('#runtime-preset');
    this.#runtimeConfigFileInput = document.querySelector('#runtime-config-file');
    this.#runtimeConfigClearBtn = document.querySelector('#runtime-config-clear');
    this.#runtimeConfigStatus = document.querySelector('#runtime-config-status');
    this.#diagnosticsSuiteSelect = document.querySelector('#diagnostics-suite');
    this.#diagnosticsRunBtn = document.querySelector('#diagnostics-run-btn');
    this.#diagnosticsVerifyBtn = document.querySelector('#diagnostics-verify-btn');
    this.#diagnosticsExportBtn = document.querySelector('#diagnostics-export-btn');
    this.#diagnosticsStatus = document.querySelector('#diagnostics-status');
    this.#diagnosticsReport = document.querySelector('#diagnostics-report');

    if (this.#diagnosticsExportBtn) {
      this.#diagnosticsExportBtn.disabled = true;
    }

    this.#workspaceImportBtn = document.querySelector('#workspace-import-btn');
    this.#workspaceRefreshBtn = document.querySelector('#workspace-refresh-btn');
    this.#workspaceStatus = document.querySelector('#workspace-status');
    this.#workspaceFiles = document.querySelector('#workspace-files');

    this.#toolSelect = document.querySelector('#tool-select');
    this.#toolArgs = document.querySelector('#tool-args');
    this.#toolRunBtn = document.querySelector('#tool-run-btn');
    this.#toolRefreshBtn = document.querySelector('#tool-refresh-btn');
    this.#toolOutput = document.querySelector('#tool-output');
    this.#toolStatus = document.querySelector('#tool-status');
  }

  #initControllers() {
    this.#capabilitiesDetector = new CapabilitiesDetector();

    this.#memoryMonitor = new MemoryMonitor(this.#memoryElements);

    this.#modelRegistry = new ModelRegistry();

    this.#modelLoader = new ModelLoader({
      onProgress: (progress) => {
        this.#progressUI?.setPhaseProgress(progress);
      },
      onSourceType: (type) => {
        this.#progressUI?.setSourceType(type);
      },
    });

    this.#modelDownloader = new ModelDownloader({
      onProgress: (modelKey, percent) => {
        this.#modelSelector?.setDownloadProgress(modelKey, percent);
      },
      onStatus: (status) => {
        if (status === 'verifying') {
          this.#setStatus('loading', 'Verifying...');
        }
      },
    });

    this.#chatController = new ChatController({
      onUserMessage: (message) => {
        this.#chatUI?.addMessage('user', message);
      },
      onGenerationStart: () => {
        this.#chatUI?.startStream();
        this.#setStatus('loading', 'Generating...');
      },
      onToken: (token) => {
        this.#chatUI?.streamToken(token);
      },
      onStats: (stats) => {
        updateStats(this.#statsElements, { tps: stats.tokensPerSec });
      },
      onGenerationComplete: (result) => {
        const uiStats = this.#chatUI?.finishStream();
        if (uiStats) {
          updateStats(this.#statsElements, { tps: uiStats.tokensPerSec });
        }
        this.#setStatus('ready', this.#modelLoader.currentModel?.name || 'Ready');
      },
      onGenerationAborted: () => {
        this.#chatUI?.cancelStream();
        this.#setStatus('ready', 'Stopped');
      },
      onGenerationError: (error) => {
        this.#chatUI?.cancelStream();
        this.#setStatus('error', 'Generation failed');
        this.#showError(`Generation failed: ${error.message}`);
      },
    });

    this.#converterController = new ConverterController({
      onStart: () => {
        if (this.#convertBtn) this.#convertBtn.disabled = true;
        if (this.#convertUrlBtn) this.#convertUrlBtn.disabled = true;
        if (this.#convertStatus) this.#convertStatus.hidden = false;
      },
      onProgress: (percent, message) => {
        if (this.#convertProgress) this.#convertProgress.style.width = `${percent}%`;
        if (this.#convertMessage) this.#convertMessage.textContent = message;
      },
      onComplete: async (modelId) => {
        await this.#modelRegistry.discover();
        this.#modelSelector?.setModels(this.#modelRegistry.getModels());
        this.#updateDiagnosticsModels();
        setTimeout(() => {
          if (this.#convertStatus) this.#convertStatus.hidden = true;
        }, 3000);
      },
      onError: (error) => {
        this.#showError(`Conversion failed: ${error.message}`);
      },
      onFinish: () => {
        if (this.#convertBtn) this.#convertBtn.disabled = false;
        if (this.#convertUrlBtn) this.#convertUrlBtn.disabled = false;
      },
    });

    this.#diagnosticsController = new DiagnosticsController({
      onSuiteStart: (suite) => {
        this.#setDiagnosticsStatus(`Running ${suite}...`);
        if (this.#diagnosticsRunBtn) this.#diagnosticsRunBtn.disabled = true;
        if (this.#diagnosticsExportBtn) this.#diagnosticsExportBtn.disabled = true;
        this.#setDiagnosticsReport('');
      },
      onSuiteComplete: (result) => {
        this.#setDiagnosticsStatus(
          `Done: ${result.suite} (${result.passed} passed, ${result.failed} failed)`
        );
        const reportPath = result.reportInfo?.path ? `Report: ${result.reportInfo.path}` : '';
        this.#setDiagnosticsReport(reportPath);
        if (this.#diagnosticsExportBtn) {
          this.#diagnosticsExportBtn.disabled = !result.report;
        }
      },
      onSuiteError: (error) => {
        this.#setDiagnosticsStatus(`Error: ${error.message}`);
        this.#showError(`Diagnostics failed: ${error.message}`);
        this.#setDiagnosticsReport('');
      },
      onSuiteFinish: () => {
        if (this.#diagnosticsRunBtn) this.#diagnosticsRunBtn.disabled = false;
      },
      onVerifyStart: () => {
        this.#setDiagnosticsStatus('Verifying...');
        if (this.#diagnosticsVerifyBtn) this.#diagnosticsVerifyBtn.disabled = true;
      },
      onVerifyComplete: (result) => {
        const status = result.valid
          ? 'Verification passed'
          : `Missing: ${result.missingShards.length}, Corrupt: ${result.corruptShards.length}`;
        this.#setDiagnosticsStatus(status);
      },
      onVerifyError: (error) => {
        this.#setDiagnosticsStatus(`Verify error: ${error.message}`);
        this.#showError(`Verify failed: ${error.message}`);
      },
      onVerifyFinish: () => {
        if (this.#diagnosticsVerifyBtn) this.#diagnosticsVerifyBtn.disabled = false;
      },
    });

    this.#quickStartController = new QuickStartController({
      onVRAMInsufficient: (required, available) => {
        this.#quickStartUI?.showVRAMBlocker(required, available);
      },
      onStorageConsent: async (modelName, required, available) => {
        const consent = await this.#quickStartUI?.showStorageConsent(modelName, required, available);
        if (consent) {
          this.#quickStartUI?.showDownloadProgress();
        }
        return consent ?? false;
      },
      onProgress: (percent, downloaded, total, speed) => {
        this.#quickStartUI?.setDownloadProgress(percent, downloaded, total, speed);
      },
      onComplete: async (modelId) => {
        this.#quickStartUI?.showReady(modelId);
        await this.#modelRegistry.discover();
        this.#modelSelector?.setModels(this.#modelRegistry.getModels());
        this.#updateDiagnosticsModels();
      },
      onDeclined: () => {
        this.#quickStartUI?.hide();
      },
      onError: (error) => {
        this.#quickStartUI?.showError(error);
      },
    });

    this.#toolingController = new ToolingController({
      toolSelect: this.#toolSelect,
      argsInput: this.#toolArgs,
      runButton: this.#toolRunBtn,
      refreshButton: this.#toolRefreshBtn,
      outputEl: this.#toolOutput,
      statusEl: this.#toolStatus,
    });
    this.#toolingController.init();


    this.#workspaceController = new WorkspaceController({
      importButton: this.#workspaceImportBtn,
      refreshButton: this.#workspaceRefreshBtn,
      statusEl: this.#workspaceStatus,
      filesEl: this.#workspaceFiles,
      onVfsReady: (vfs) => this.#toolingController?.setVfs(vfs),
    });
  }

  #initUIComponents() {
    const container = document.querySelector('#app');

    this.#modelSelector = new ModelSelector(container, {
      onSelect: (model, opts) => this.selectModel(model, opts),
      onDownload: (model, opts) => this.downloadModel(model, opts),
      onDelete: (model) => this.deleteModel(model),
      onQuickStart: (model) => {
        const modelId = model.sources?.remote?.id || model.key;
        this.startQuickStart(modelId);
      },
    });

    this.#chatUI = new ChatUI(container, {
      onSend: (message) => this.chat(message),
      onStop: () => this.stopGeneration(),
      onClear: () => this.clearConversation(),
    });

    this.#progressUI = new ProgressUI(container);

    this.#quickStartUI = new QuickStartUI(container, {
      onDownloadComplete: (modelId) => log.debug('QuickStart', `Download complete: ${modelId}`),
      onRunModel: (modelId) => this.#runQuickStartModel(modelId),
      onCancel: () => log.debug('QuickStart', 'Cancelled by user'),
    });
  }

  #initEventListeners() {
    // Sampling inputs
    if (this.#temperatureInput) {
      this.#temperatureInput.addEventListener('change', () =>
        clampInputValue(this.#temperatureInput, 0.1, 2.0)
      );
    }
    if (this.#topPInput) {
      this.#topPInput.addEventListener('change', () =>
        clampInputValue(this.#topPInput, 0, 1)
      );
    }
    if (this.#topKInput) {
      this.#topKInput.addEventListener('change', () =>
        clampInputValue(this.#topKInput, 0, 200)
      );
    }

    // Convert button
    if (this.#convertBtn) {
      if (ConverterController.isSupported()) {
        this.#convertBtn.addEventListener('click', () => {
          const converterConfig = this.#getConverterConfig();
          this.#converterController.convert(converterConfig ? { converterConfig } : {});
        });
      } else {
        this.#convertBtn.disabled = true;
        this.#convertBtn.title = 'Model conversion requires File System Access API (Chrome/Edge)';
      }
    }
    if (this.#convertUrlBtn) {
      if (ConverterController.isSupported()) {
        this.#convertUrlBtn.addEventListener('click', () => this.#convertFromUrls());
      } else {
        this.#convertUrlBtn.disabled = true;
        this.#convertUrlBtn.title = 'Model conversion requires File System Access API (Chrome/Edge)';
      }
    }

    if (this.#runtimeConfigFileInput) {
      this.#runtimeConfigFileInput.addEventListener('change', (event) => {
        const file = event.target?.files?.[0];
        if (!file) return;
        this.#loadRuntimeConfigFile(file);
      });
    }
    if (this.#runtimePresetSelect) {
      this.#runtimePresetSelect.addEventListener('change', () => this.#updateRuntimeConfigStatus());
    }
    if (this.#runtimeConfigClearBtn) {
      this.#runtimeConfigClearBtn.addEventListener('click', () => this.#clearRuntimeConfigOverride());
    }
    if (this.#diagnosticsRunBtn) {
      this.#diagnosticsRunBtn.addEventListener('click', () => this.#runDiagnosticsSuite());
    }
    if (this.#diagnosticsVerifyBtn) {
      this.#diagnosticsVerifyBtn.addEventListener('click', () => this.#verifyDiagnosticsModel());
    }
    if (this.#diagnosticsExportBtn) {
      this.#diagnosticsExportBtn.addEventListener('click', () => this.#exportDiagnosticsReport());
    }

    // Memory control buttons
    if (this.#unloadModelBtn) {
      this.#unloadModelBtn.addEventListener('click', () => this.#unloadCurrentModel());
    }
    if (this.#clearMemoryBtn) {
      this.#clearMemoryBtn.addEventListener('click', () => this.#clearAllMemory());
    }
  }

  // ============================================================================
  // Public API
  // ============================================================================

  async selectModel(model, opts = {}) {
    if (this.#chatController.isGenerating) {
      this.#showError('Cannot switch models while generating');
      return;
    }

    const modelInfo = typeof model === 'string'
      ? this.#modelRegistry.findByKey(model)
      : model;

    if (!modelInfo) {
      this.#showError(`Unknown model: ${model}`);
      return;
    }

    if (!this.#modelRegistry.isAvailableLocally(modelInfo)) {
      this.#showError('Model not available locally. Download it first.');
      return;
    }

    this.#setStatus('loading', 'Loading model...');
    this.#progressUI?.show('Loading model...');

    try {
      await this.#modelLoader.load(modelInfo, { preferredSource: opts.preferredSource });

      this.#modelSelector?.setActiveModel(modelInfo.key);
      this.#updateDiagnosticsModels();
      this.#progressUI?.hide();
      this.#setStatus('ready', `${modelInfo.name} loaded`);
      this.#chatUI?.setInputEnabled(true);
      this.#chatUI?.focusInput();
      this.#updateInitialStats();

      log.info('App', `Model loaded: ${modelInfo.name} (${modelInfo.key})`);
    } catch (error) {
      log.error('App', 'Model load failed:', error);
      this.#progressUI?.hide();
      this.#setStatus('error', 'Load failed');
      this.#showError(`Failed to load model: ${error.message}`);
    }
  }

  async downloadModel(model, opts = {}) {
    this.#setStatus('loading', `Downloading ${model.name}...`);

    try {
      await this.#modelDownloader.download(model);
      this.#setStatus('ready', 'Download complete');
      await this.#modelRegistry.discover();
      this.#modelSelector?.setModels(this.#modelRegistry.getModels());
      this.#updateDiagnosticsModels();

      if (opts.runAfter) {
        const updatedModel = this.#modelRegistry.findByKey(model.key);
        if (updatedModel) {
          await this.selectModel(updatedModel);
        }
      }
    } catch (error) {
      this.#setStatus('error', 'Download failed');
      this.#showError(`Download failed: ${error.message}`);
    }
  }

  async deleteModel(model) {
    try {
      // Unload if currently active
      if (this.#modelLoader.currentModel?.key === model.key) {
        await this.#modelLoader.unload();
        this.#modelSelector?.setActiveModel(null);
        this.#chatUI?.setInputEnabled(false);
      }

      await this.#modelDownloader.delete(model);
      this.#setStatus('ready', 'Cache cleared');
      await this.#modelRegistry.discover();
      this.#modelSelector?.setModels(this.#modelRegistry.getModels());
      this.#updateDiagnosticsModels();
    } catch (error) {
      log.error('App', 'Delete failed:', error);
      this.#showError(`Delete failed: ${error.message}`);
    }
  }

  async chat(message) {
    if (!this.#modelLoader.currentModel) {
      this.#showError('No model loaded');
      return;
    }

    // Update sampling params from UI
    this.#chatController.setSamplingParams({
      temperature: this.#getSamplingValue(this.#temperatureInput, 0.7),
      topP: this.#getSamplingValue(this.#topPInput, 0.9),
      topK: this.#getSamplingValue(this.#topKInput, 40, true),
    });

    try {
      await this.#chatController.generate(message, this.#modelLoader.pipeline);
    } catch (error) {
      // Error already handled by controller callbacks
    }
  }

  stopGeneration() {
    this.#chatController.stop();
  }

  clearConversation() {
    this.#chatController.clear(this.#modelLoader.pipeline);
    this.#chatUI?.clear();
  }

  async startQuickStart(modelId) {
    await this.#quickStartController.start(modelId);
  }

  getStatus() {
    return {
      model: this.#modelLoader.currentModel?.key || null,
      modelName: this.#modelLoader.currentModel?.name || null,
      isGenerating: this.#chatController.isGenerating,
      capabilities: this.#capabilitiesDetector.getState(),
      memory: this.#modelLoader.getMemoryStats(),
    };
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  #setStatus(state, text) {
    setStatus(this.#statusDot, this.#statusText, state, text);
  }

  #showError(message) {
    showError(
      document.querySelector('#error-modal'),
      document.querySelector('#error-message'),
      document.querySelector('#error-close'),
      message
    );
  }

  #getSamplingValue(input, defaultValue, isInt = false) {
    const value = isInt
      ? parseInt(input?.value || '', 10)
      : parseFloat(input?.value || '');
    return Number.isFinite(value) ? value : defaultValue;
  }

  #updateInitialStats() {
    updateStats(this.#statsElements, { tps: 'Ready' });

    const heap = performance.memory?.usedJSHeapSize;
    if (heap) {
      updateStats(this.#statsElements, { memoryMB: Math.round(heap / 1024 / 1024) });
    }

    const kvStats = this.#modelLoader.getKVCacheStats();
    if (kvStats) {
      updateStats(this.#statsElements, {
        kvSeqLen: kvStats.seqLen,
        kvMaxLen: kvStats.maxSeqLen,
      });
    }

    if (this.#unloadModelBtn) {
      this.#unloadModelBtn.disabled = false;
    }
  }

  #setDiagnosticsStatus(message) {
    if (this.#diagnosticsStatus) {
      this.#diagnosticsStatus.textContent = message;
    }
  }

  #setDiagnosticsReport(message) {
    if (this.#diagnosticsReport) {
      this.#diagnosticsReport.textContent = message;
    }
  }

  #populateConverterPresets() {
    if (!this.#convertPresetSelect) return;
    const presets = listPresets().slice().sort();
    this.#convertPresetSelect.innerHTML = '';

    const autoOption = document.createElement('option');
    autoOption.value = '';
    autoOption.textContent = 'auto-detect';
    this.#convertPresetSelect.appendChild(autoOption);

    for (const preset of presets) {
      const option = document.createElement('option');
      option.value = preset;
      option.textContent = preset;
      this.#convertPresetSelect.appendChild(option);
    }
  }

  #syncConverterUIFromConfig() {
    const runtime = getRuntimeConfig();
    const overrides = runtime?.shared?.tooling?.converter;
    const config = createConverterConfig(overrides && typeof overrides === 'object' ? overrides : {});

    if (this.#convertPresetSelect && config.presets?.model) {
      const presetId = config.presets.model;
      const hasPreset = Array.from(this.#convertPresetSelect.options)
        .some((option) => option.value === presetId);
      if (!hasPreset) {
        const option = document.createElement('option');
        option.value = presetId;
        option.textContent = presetId;
        this.#convertPresetSelect.appendChild(option);
      }
      this.#convertPresetSelect.value = presetId;
    }
    if (this.#convertAllowDownloadInput) {
      this.#convertAllowDownloadInput.checked = config.http?.allowDownloadFallback !== false;
    }
    if (this.#convertMaxDownloadInput) {
      const maxBytes = config.http?.maxDownloadBytes;
      const maxValue = Number.isFinite(maxBytes) && maxBytes > 0 ? Math.round(maxBytes / MB) : '';
      this.#convertMaxDownloadInput.value = maxValue ? String(maxValue) : '';
    }
  }

  #updateRuntimeConfigStatus() {
    if (!this.#runtimeConfigStatus) return;
    if (this.#diagnosticsRuntimeConfig) {
      const label = this.#diagnosticsRuntimeLabel || 'runtime.json';
      this.#runtimeConfigStatus.textContent = `File: ${label}`;
    } else {
      const preset = this.#runtimePresetSelect?.value || 'default';
      this.#runtimeConfigStatus.textContent = `Preset: ${preset}`;
    }
    if (this.#runtimePresetSelect) {
      this.#runtimePresetSelect.disabled = Boolean(this.#diagnosticsRuntimeConfig);
    }
    if (this.#runtimeConfigClearBtn) {
      this.#runtimeConfigClearBtn.disabled = !this.#diagnosticsRuntimeConfig;
    }
  }

  #resolveRuntimeFromConfig(config) {
    if (!config || typeof config !== 'object') return null;
    if (config.runtime && typeof config.runtime === 'object') return config.runtime;
    if (config.shared || config.loading || config.inference || config.emulation) return config;
    return null;
  }

  async #loadRuntimeConfigFile(file) {
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      const runtime = this.#resolveRuntimeFromConfig(parsed);
      if (!runtime) {
        throw new Error('Runtime config is missing runtime fields.');
      }
      this.#diagnosticsRuntimeConfig = runtime;
      this.#diagnosticsRuntimeLabel = file.name;
      this.#updateRuntimeConfigStatus();
    } catch (error) {
      this.#clearRuntimeConfigOverride();
      this.#showError(`Failed to load runtime config: ${error.message}`);
    }
  }

  #clearRuntimeConfigOverride() {
    this.#diagnosticsRuntimeConfig = null;
    this.#diagnosticsRuntimeLabel = null;
    if (this.#runtimeConfigFileInput) {
      this.#runtimeConfigFileInput.value = '';
    }
    this.#updateRuntimeConfigStatus();
  }

  #exportDiagnosticsReport() {
    try {
      this.#diagnosticsController?.exportLastReport();
    } catch (error) {
      this.#showError(error.message);
    }
  }

  #getConverterConfig() {
    const runtime = getRuntimeConfig();
    const overrides = runtime?.shared?.tooling?.converter;
    const config = createConverterConfig(overrides && typeof overrides === 'object' ? overrides : {});

    const presetOverride = this.#convertPresetSelect?.value || '';
    config.presets.model = presetOverride || null;

    if (this.#convertAllowDownloadInput) {
      config.http.allowDownloadFallback = this.#convertAllowDownloadInput.checked;
    }

    const maxValue = parseInt(this.#convertMaxDownloadInput?.value || '', 10);
    if (Number.isFinite(maxValue)) {
      config.http.maxDownloadBytes = Math.max(1, Math.floor(maxValue)) * MB;
    } else {
      config.http.maxDownloadBytes = null;
    }

    return config;
  }

  #updateDiagnosticsModels() {
    if (!this.#diagnosticsModelSelect) return;
    const models = this.#modelRegistry.getModels();
    const activeKey = this.#modelLoader.currentModel?.key || null;
    const current = this.#diagnosticsModelSelect.value || activeKey || '';

    this.#diagnosticsModelSelect.innerHTML = '';

    for (const model of models) {
      const option = document.createElement('option');
      option.value = model.key;
      const sourceLabel = model.sources?.browser
        ? 'cache'
        : (model.sources?.server ? 'server' : (model.sources?.remote ? 'remote' : 'unknown'));
      option.textContent = `${model.name} (${sourceLabel})`;
      this.#diagnosticsModelSelect.appendChild(option);
    }

    if (current) {
      this.#diagnosticsModelSelect.value = current;
    }
  }

  #getDiagnosticsModel() {
    const key = this.#diagnosticsModelSelect?.value;
    if (!key) return null;
    return this.#modelRegistry.findByKey(key);
  }

  #parseUrlList(value) {
    return String(value || '')
      .split(/\\r?\\n|,/)
      .map((entry) => entry.trim())
      .filter(Boolean);
  }

  async #convertFromUrls() {
    const urls = this.#parseUrlList(this.#convertUrlInput?.value);
    if (urls.length === 0) {
      this.#showError('Provide at least one URL to convert.');
      return;
    }

    try {
      const converterConfig = this.#getConverterConfig();
      await this.#converterController.convertRemote(urls, converterConfig ? { converterConfig } : {});
    } catch (error) {
      this.#showError(`Conversion failed: ${error.message}`);
    }
  }

  async #runDiagnosticsSuite() {
    const suite = this.#diagnosticsSuiteSelect?.value || 'inference';
    const model = this.#getDiagnosticsModel();

    if (suite !== 'kernels' && !model) {
      this.#showError('Select a model to run diagnostics.');
      return;
    }

    const runtimePreset = this.#diagnosticsRuntimeConfig ? null : (this.#runtimePresetSelect?.value || null);
    const runtimeConfig = this.#diagnosticsRuntimeConfig;

    try {
      await this.#diagnosticsController.runSuite(model, {
        suite,
        runtimePreset,
        runtimeConfig,
      });
    } catch (error) {
      this.#showError(`Diagnostics failed: ${error.message}`);
    }
  }

  async #verifyDiagnosticsModel() {
    const model = this.#getDiagnosticsModel();
    if (!model) {
      this.#showError('Select a model to verify.');
      return;
    }
    if (!model.sources?.browser?.id) {
      this.#showError('Verify requires a cached (browser) model.');
      return;
    }

    try {
      await this.#diagnosticsController.verifyModel(model);
    } catch (error) {
      this.#showError(`Verify failed: ${error.message}`);
    }
  }

  async #unloadCurrentModel() {
    if (!this.#modelLoader.pipeline) return;

    this.#setStatus('loading', 'Unloading model...');

    try {
      await this.#modelLoader.unload();
      this.#modelSelector?.setActiveModel(null);
      this.#chatUI?.setInputEnabled(false);
      resetStats(this.#statsElements);

      if (this.#unloadModelBtn) {
        this.#unloadModelBtn.disabled = true;
      }

      this.#setStatus('ready', 'Model unloaded');
      this.#memoryMonitor.update();
    } catch (error) {
      log.error('App', 'Unload failed:', error);
      this.#setStatus('error', 'Unload failed');
    }
  }

  async #clearAllMemory() {
    this.#setStatus('loading', 'Clearing memory...');

    try {
      await this.#modelLoader.clearAllMemory();
      this.#modelSelector?.setActiveModel(null);
      this.#chatUI?.setInputEnabled(false);
      resetStats(this.#statsElements);

      if (this.#unloadModelBtn) {
        this.#unloadModelBtn.disabled = true;
      }

      this.#setStatus('ready', 'Memory cleared');
      this.#memoryMonitor.update();
    } catch (error) {
      log.error('App', 'Clear memory failed:', error);
      this.#setStatus('error', 'Clear failed');
    }
  }

  async #runQuickStartModel(modelId) {
    log.info('QuickStart', `Running model ${modelId}`);

    let model = this.#modelRegistry.findByKey(modelId) ||
                this.#modelRegistry.findByBrowserId(modelId);

    if (!model) {
      await this.#modelRegistry.discover();
      this.#modelSelector?.setModels(this.#modelRegistry.getModels());
      this.#updateDiagnosticsModels();
      model = this.#modelRegistry.findByKey(modelId) ||
              this.#modelRegistry.findByBrowserId(modelId);
    }

    if (model) {
      await this.selectModel(model);
    } else {
      this.#showError(`Model ${modelId} not found after download`);
    }
  }
}

// ============================================================================
// Initialize on DOM ready
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  const app = new DopplerDemo();
  app.init().catch((err) => log.error('App', 'Initialization failed', err));

  // Expose for debugging
  window.dopplerDemo = app;
});

export default DopplerDemo;
