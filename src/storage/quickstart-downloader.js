

import { downloadModel } from './downloader.js';
import {
  runPreflightChecks,
  MODEL_REQUIREMENTS,
} from './preflight.js';
import { formatBytes } from './quota.js';
import { getCdnBasePath } from './download-types.js';
import quickstartRegistry from '../client/doppler-registry.json' with { type: 'json' };
import { buildHfResolveBaseUrl, DEFAULT_HF_CDN_BASE_URL } from '../utils/hf-resolve-url.js';

// ============================================================================
// Model Registry
// ============================================================================


let cdnBaseOverride = null;

export function setCDNBaseUrl(url) {
  const normalized = typeof url === 'string' ? url.trim().replace(/\/$/, '') : '';
  cdnBaseOverride = normalized || null;
}


export function getCDNBaseUrl() {
  return cdnBaseOverride ?? getCdnBasePath() ?? DEFAULT_HF_CDN_BASE_URL;
}


const QUICKSTART_DISPLAY_NAMES = {
  'gemma-3-270m-it-q4k-ehf16-af32': 'Gemma 3 270M IT (Q4K)',
  'google-embeddinggemma-300m-q4k-ehf16-af32': 'EmbeddingGemma 300M (Q4K)',
  'gemma-3-1b-it-q4k-ehf16-af32': 'Gemma 3 1B IT (Q4K)',
  'gemma-4-e2b-it-q4k-ehf16-af32': 'Gemma 4 E2B IT (Q4K)',
  'gemma-4-e2b-it-q4k-ehf16-af32-int4ple': 'Gemma 4 E2B IT (Q4K + INT4 PLE)',
  'qwen-3-5-0-8b-q4k-ehaf16': 'Qwen 3.5 0.8B (Q4K)',
  'qwen-3-5-2b-q4k-ehaf16': 'Qwen 3.5 2B (Q4K)',
};

function normalizeRegistryText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function buildQuickStartModelsFromRegistry(registry) {
  const rows = Array.isArray(registry?.models) ? registry.models : [];
  const models = {};
  for (const row of rows) {
    const modelId = normalizeRegistryText(row?.modelId);
    const displayName = QUICKSTART_DISPLAY_NAMES[modelId];
    const requirements = MODEL_REQUIREMENTS[modelId];
    const hf = row?.hf && typeof row.hf === 'object' ? row.hf : null;
    const repoId = normalizeRegistryText(hf?.repoId);
    const revision = normalizeRegistryText(hf?.revision);
    const repoPath = normalizeRegistryText(hf?.path);
    if (!modelId || !displayName || !requirements || !repoId || !revision || !repoPath) {
      throw new Error(`Quickstart registry entry "${modelId || 'unknown'}" is incomplete for storage download.`);
    }
    models[modelId] = {
      modelId,
      displayName,
      baseUrl: null,
      hf: {
        repoId,
        revision,
        path: repoPath,
      },
      requirements,
    };
  }
  return models;
}

export const QUICKSTART_MODELS = buildQuickStartModelsFromRegistry(quickstartRegistry);


export function getQuickStartModel(modelId) {
  return QUICKSTART_MODELS[modelId];
}


export function listQuickStartModels() {
  return Object.values(QUICKSTART_MODELS);
}


export function registerQuickStartModel(config) {
  QUICKSTART_MODELS[config.modelId] = config;
}

function resolveQuickStartModelBaseUrl(config) {
  if (typeof config?.baseUrl === 'string' && config.baseUrl.trim().length > 0) {
    return config.baseUrl.trim().replace(/\/$/, '');
  }
  if (config?.hf) {
    return buildHfResolveBaseUrl(config.hf, { cdnBasePath: getCDNBaseUrl() });
  }
  throw new Error(
    `Quickstart model "${config?.modelId ?? 'unknown'}" is missing an explicit baseUrl or hosted Hugging Face source.`
  );
}

// ============================================================================
// Download Functions
// ============================================================================


export async function downloadQuickStartModel(
  modelId,
  options = {}
) {
  const config = QUICKSTART_MODELS[modelId];

  if (!config) {
    return {
      success: false,
      modelId,
      error: `Unknown model: ${modelId}. Available: ${Object.keys(QUICKSTART_MODELS).join(', ')}`,
    };
  }

  const {
    onProgress,
    onPreflightComplete,
    onStorageConsent,
    signal,
    concurrency = 3,
    skipPreflight = false,
  } = options;

  // -------------------------------------------------------------------------
  // Step 1: Pre-flight checks
  // -------------------------------------------------------------------------
  
  let preflight;

  if (!skipPreflight) {
    try {
      preflight = await runPreflightChecks(config.requirements);
      onPreflightComplete?.(preflight);

      if (!preflight.canProceed) {
        return {
          success: false,
          modelId,
          error: preflight.blockers.join('; '),
          preflight,
          blockedByPreflight: true,
        };
      }
    } catch (err) {
      return {
        success: false,
        modelId,
        error: `Preflight check failed: ${ (err).message}`,
      };
    }
  }

  // -------------------------------------------------------------------------
  // Step 2: Request user consent
  // -------------------------------------------------------------------------
  if (onStorageConsent) {
    const requiredBytes = config.requirements.downloadSize;
    const availableBytes = preflight?.storage.available ?? 0;

    try {
      const consent = await onStorageConsent(requiredBytes, availableBytes, config.displayName);

      if (!consent) {
        return {
          success: false,
          modelId,
          error: 'User declined storage consent',
          preflight,
          userDeclined: true,
        };
      }
    } catch (err) {
      return {
        success: false,
        modelId,
        error: `Consent flow failed: ${ (err).message}`,
        preflight,
      };
    }
  }

  // -------------------------------------------------------------------------
  // Step 3: Download model
  // -------------------------------------------------------------------------
  try {
    // Check for abort before starting
    if (signal?.aborted) {
      return {
        success: false,
        modelId,
        error: 'Download aborted',
        preflight,
      };
    }

    
    const downloadOpts = {
      concurrency,
      requestPersist: true,
      modelId: config.modelId,
      signal,
    };

    const baseUrl = resolveQuickStartModelBaseUrl(config);
    const success = await downloadModel(
      baseUrl,
      onProgress,
      downloadOpts
    );

    if (!success) {
      return {
        success: false,
        modelId,
        error: 'Download failed',
        preflight,
      };
    }

    return {
      success: true,
      modelId,
      preflight,
    };
  } catch (err) {
    const errorMessage =  (err).message;

    // Handle specific error types
    if (errorMessage.includes('aborted') || signal?.aborted) {
      return {
        success: false,
        modelId,
        error: 'Download aborted by user',
        preflight,
      };
    }

    if (errorMessage.includes('quota') || errorMessage.includes('storage')) {
      return {
        success: false,
        modelId,
        error: `Storage error: ${errorMessage}`,
        preflight,
      };
    }

    return {
      success: false,
      modelId,
      error: `Download failed: ${errorMessage}`,
      preflight,
    };
  }
}


export async function isModelDownloaded(modelId) {
  // Import dynamically to avoid circular deps
  const { modelExists } = await import('./shard-manager.js');
  return modelExists(modelId);
}


export function getModelDownloadSize(modelId) {
  const config = QUICKSTART_MODELS[modelId];
  return config?.requirements.downloadSize ?? null;
}


export function formatModelInfo(modelId) {
  const config = QUICKSTART_MODELS[modelId];
  if (!config) return null;

  const { requirements } = config;
  return [
    config.displayName,
    `${requirements.paramCount} parameters`,
    `${requirements.quantization} quantization`,
    `${formatBytes(requirements.downloadSize)} download`,
    `${formatBytes(requirements.vramRequired)} VRAM required`,
  ].join(' | ');
}
