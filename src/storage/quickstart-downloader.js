

import { downloadModel } from './downloader.js';
import {
  runPreflightChecks,
  MODEL_REQUIREMENTS,
} from './preflight.js';
import { formatBytes } from './quota.js';
import { getCdnBasePath } from './download-types.js';
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


export const QUICKSTART_MODELS = {
  'gemma-3-270m-it-q4k-ehf16-af32': {
    modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
    displayName: 'Gemma 3 270M IT (Q4K)',
    baseUrl: null,
    hf: {
      repoId: 'Clocksmith/rdrr',
      revision: 'f242e44c9a35e72186aa61e4cc0c9873f0596741',
      path: 'models/gemma-3-270m-it-q4k-ehf16-af32',
    },
    requirements: MODEL_REQUIREMENTS['gemma-3-270m-it-q4k-ehf16-af32'],
  },
  'google-embeddinggemma-300m-q4k-ehf16-af32': {
    modelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
    displayName: 'EmbeddingGemma 300M (Q4K)',
    baseUrl: null,
    hf: {
      repoId: 'Clocksmith/rdrr',
      revision: 'f242e44c9a35e72186aa61e4cc0c9873f0596741',
      path: 'models/google-embeddinggemma-300m-q4k-ehf16-af32',
    },
    requirements: MODEL_REQUIREMENTS['google-embeddinggemma-300m-q4k-ehf16-af32'],
  },
  'gemma-3-1b-it-q4k-ehf16-af32': {
    modelId: 'gemma-3-1b-it-q4k-ehf16-af32',
    displayName: 'Gemma 3 1B IT (Q4K)',
    baseUrl: null,
    hf: {
      repoId: 'Clocksmith/rdrr',
      revision: 'f242e44c9a35e72186aa61e4cc0c9873f0596741',
      path: 'models/gemma-3-1b-it-q4k-ehf16-af32',
    },
    requirements: MODEL_REQUIREMENTS['gemma-3-1b-it-q4k-ehf16-af32'],
  },
  'qwen-3-5-0-8b-q4k-ehaf16': {
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    displayName: 'Qwen 3.5 0.8B (Q4K)',
    baseUrl: null,
    hf: {
      repoId: 'Clocksmith/rdrr',
      revision: 'f242e44c9a35e72186aa61e4cc0c9873f0596741',
      path: 'models/qwen-3-5-0-8b-q4k-ehaf16',
    },
    requirements: MODEL_REQUIREMENTS['qwen-3-5-0-8b-q4k-ehaf16'],
  },
  'qwen-3-5-2b-q4k-ehaf16': {
    modelId: 'qwen-3-5-2b-q4k-ehaf16',
    displayName: 'Qwen 3.5 2B (Q4K)',
    baseUrl: null,
    hf: {
      repoId: 'Clocksmith/rdrr',
      revision: 'f242e44c9a35e72186aa61e4cc0c9873f0596741',
      path: 'models/qwen-3-5-2b-q4k-ehaf16',
    },
    requirements: MODEL_REQUIREMENTS['qwen-3-5-2b-q4k-ehaf16'],
  },
};


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
