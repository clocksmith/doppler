

import { downloadModel } from './downloader.js';
import {
  runPreflightChecks,
  GEMMA_1B_REQUIREMENTS,
} from './preflight.js';
import { formatBytes } from './quota.js';
import { getCdnBasePath } from './download-types.js';

// ============================================================================
// Model Registry
// ============================================================================


let cdnBaseOverride = null;


function getEffectiveCDNBaseUrl() {
  const runtimeBase = getCdnBasePath();
  const base = cdnBaseOverride ?? runtimeBase ?? '';
  if (base) return base;

  // Auto-detect: use same origin for Firebase Hosting or local dev
  if (typeof window !== 'undefined') {
    return `${window.location.origin}/doppler/models`;
  }
  // Fallback for Node.js/SSR
  return '/doppler/models';
}


export function setCDNBaseUrl(url) {
  cdnBaseOverride = url.replace(/\/$/, ''); // Remove trailing slash
}


export function getCDNBaseUrl() {
  return getEffectiveCDNBaseUrl();
}


export const QUICKSTART_MODELS = {
  'gemma-3-1b-it-q4': {
    modelId: 'gemma-3-1b-it-q4',
    displayName: 'Gemma 3 1B IT (Q4)',
    baseUrl: 'https://huggingface.co/clocksmith/gemma3-1b-rdrr/resolve/main',
    requirements: GEMMA_1B_REQUIREMENTS,
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

    const success = await downloadModel(
      config.baseUrl,
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
