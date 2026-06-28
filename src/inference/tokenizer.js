

import { log } from '../debug/index.js';
import { BaseTokenizer } from './tokenizers/base.js';
import { TransformersTokenizer, BundledTokenizer } from './tokenizers/bundled.js';
import { SentencePieceTokenizer } from './tokenizers/sentencepiece.js';
import { BPETokenizer } from './tokenizers/bpe.js';
import { toArrayBuffer } from '../utils/array-buffer.js';

const BUNDLED_TOKENIZER_CACHE = new Map();
const TOKENIZER_LOAD_PHASES = Object.freeze([
  'configResolution',
  'cacheLookup',
  'backendCreate',
  'assetLoad',
  'assetParse',
  'backendLoad',
  'cacheStore',
]);

function hasScheme(value) {
  return /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(value);
}

function parseTokenizerJsonPayload(value) {
  if (value == null) {
    return null;
  }
  if (typeof value === 'string') {
    return JSON.parse(value);
  }
  if (typeof value === 'object') {
    return value;
  }
  throw new Error('Tokenizer JSON loader must return an object, JSON string, or null.');
}

function roundTokenizerTimingMs(value) {
  return Number.isFinite(value) ? Number(value.toFixed(3)) : null;
}

function createTokenizerLoadTiming(modelId) {
  return {
    schemaVersion: 1,
    source: 'doppler-tokenizer',
    modelId: typeof modelId === 'string' ? modelId : null,
    status: 'running',
    tokenizerType: null,
    tokenizerFile: null,
    backend: null,
    assetSource: null,
    cacheHit: false,
    phasesMs: Object.fromEntries(TOKENIZER_LOAD_PHASES.map((phase) => [phase, null])),
    totalMs: null,
    error: null,
  };
}

function cloneTokenizerLoadTiming(loadTiming) {
  if (!loadTiming || typeof loadTiming !== 'object') {
    return null;
  }
  return JSON.parse(JSON.stringify(loadTiming));
}

function finishTokenizerLoadPhase(loadTiming, phase, startMs) {
  if (!loadTiming?.phasesMs || !phase) return;
  const elapsedMs = roundTokenizerTimingMs(performance.now() - startMs);
  const currentMs = loadTiming.phasesMs[phase];
  loadTiming.phasesMs[phase] = Number.isFinite(currentMs)
    ? roundTokenizerTimingMs(currentMs + elapsedMs)
    : elapsedMs;
}

async function timedTokenizerLoadPhase(loadTiming, phase, run) {
  const startMs = performance.now();
  try {
    return await run();
  } finally {
    finishTokenizerLoadPhase(loadTiming, phase, startMs);
  }
}

function timedTokenizerLoadPhaseSync(loadTiming, phase, run) {
  const startMs = performance.now();
  try {
    return run();
  } finally {
    finishTokenizerLoadPhase(loadTiming, phase, startMs);
  }
}

function finishTokenizerLoadTiming(loadTiming, status, startMs, error = null) {
  loadTiming.status = status;
  loadTiming.totalMs = roundTokenizerTimingMs(performance.now() - startMs);
  loadTiming.error = error == null
    ? null
    : (error instanceof Error ? error.message : String(error));
}

function buildBundledTokenizerCacheKey(modelId, tokenizerConfig) {
  const tokenizerType = typeof tokenizerConfig?.type === 'string'
    ? tokenizerConfig.type.trim().toLowerCase()
    : '';
  const tokenizerFile = typeof tokenizerConfig?.file === 'string'
    ? tokenizerConfig.file.trim()
    : '';
  if (!tokenizerType || !tokenizerFile) {
    return null;
  }
  return JSON.stringify({
    modelId: String(modelId || '').trim() || 'unknown',
    tokenizerType,
    tokenizerFile,
    vocabSize: Number.isFinite(tokenizerConfig?.vocabSize) ? tokenizerConfig.vocabSize : null,
    eosToken: tokenizerConfig?.eosToken ?? null,
    bosToken: tokenizerConfig?.bosToken ?? null,
    padToken: tokenizerConfig?.padToken ?? null,
    unkToken: tokenizerConfig?.unkToken ?? null,
  });
}

export class Tokenizer {
  
  backend = null;

  
  config = null;

  loadTiming = null;

  
  async initialize(manifest, options = {}) {
    const loadStart = performance.now();
    const modelId = manifest?.modelId || 'unknown';
    const loadTiming = createTokenizerLoadTiming(modelId);
    this.loadTiming = loadTiming;
    try {
      const tokenizerConfig = timedTokenizerLoadPhaseSync(loadTiming, 'configResolution', () => {
        const tokenizerHints = options.tokenizerHints || {};
        const resolvedTokenizerConfig = { ...tokenizerHints, ...(manifest.tokenizer || {}) };
        const eosTokenId = Array.isArray(manifest.eos_token_id)
          ? manifest.eos_token_id[0]
          : manifest.eos_token_id;

        if (resolvedTokenizerConfig.eosToken == null && Array.isArray(resolvedTokenizerConfig.eosTokens) && resolvedTokenizerConfig.eosTokens.length > 0) {
          resolvedTokenizerConfig.eosToken = resolvedTokenizerConfig.eosTokens[0];
        }
        if (resolvedTokenizerConfig.eosToken == null && typeof eosTokenId === 'number') {
          resolvedTokenizerConfig.eosToken = eosTokenId;
        }
        if (resolvedTokenizerConfig.bosToken == null && resolvedTokenizerConfig.bosTokenId != null) {
          resolvedTokenizerConfig.bosToken = resolvedTokenizerConfig.bosTokenId;
        }
        if (resolvedTokenizerConfig.padToken == null && resolvedTokenizerConfig.padTokenId != null) {
          resolvedTokenizerConfig.padToken = resolvedTokenizerConfig.padTokenId;
        }
        if (resolvedTokenizerConfig.unkToken == null && resolvedTokenizerConfig.unkTokenId != null) {
          resolvedTokenizerConfig.unkToken = resolvedTokenizerConfig.unkTokenId;
        }
        return resolvedTokenizerConfig;
      });
      loadTiming.tokenizerType = typeof tokenizerConfig.type === 'string' ? tokenizerConfig.type : null;
      loadTiming.tokenizerFile = typeof tokenizerConfig.file === 'string' ? tokenizerConfig.file : null;

      // Check for bundled or HuggingFace tokenizer first (eliminates transformers.js dependency)
      const isBundled = tokenizerConfig.type === 'bundled' || tokenizerConfig.type === 'huggingface';
      if (isBundled && tokenizerConfig.file) {
        loadTiming.backend = 'bundled';
        tokenizerConfig.deferSpecialTokens = true;
        log.info('Tokenizer', `Loading ${tokenizerConfig.type} tokenizer from ${tokenizerConfig.file}`);
        const { cacheKey, cachedBackend } = timedTokenizerLoadPhaseSync(loadTiming, 'cacheLookup', () => {
          const resolvedCacheKey = buildBundledTokenizerCacheKey(modelId, tokenizerConfig);
          return {
            cacheKey: resolvedCacheKey,
            cachedBackend: resolvedCacheKey ? BUNDLED_TOKENIZER_CACHE.get(resolvedCacheKey) : null,
          };
        });
        loadTiming.cacheHit = Boolean(cachedBackend);
        if (cachedBackend) {
          log.info('Tokenizer', `Bundled tokenizer cache hit: ${tokenizerConfig.file}`);
          this.backend = cachedBackend;
          this.config = tokenizerConfig;
          loadTiming.assetSource = 'cache';
          finishTokenizerLoadTiming(loadTiming, 'complete', loadStart);
          return;
        }

        this.backend = timedTokenizerLoadPhaseSync(
          loadTiming,
          'backendCreate',
          () => new BundledTokenizer(tokenizerConfig)
        );

        const baseUrl = options.baseUrl;

        let tokenizerJson = null;
        const attemptedPaths = [];

        // Try to load tokenizer.json
        if (typeof options.loadTokenizerJson === 'function') {
          attemptedPaths.push('custom-loader');
          try {
            const loaded = await timedTokenizerLoadPhase(
              loadTiming,
              'assetLoad',
              () => options.loadTokenizerJson()
            );
            tokenizerJson = timedTokenizerLoadPhaseSync(
              loadTiming,
              'assetParse',
              () => parseTokenizerJsonPayload(loaded)
            );
            if (tokenizerJson) {
              loadTiming.assetSource = 'custom-loader';
            }
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            log.warn('Tokenizer', `Failed to load bundled tokenizer from custom loader: ${message}`);
          }
        }

        if (!tokenizerJson && baseUrl) {
          // Load from remote URL
          const tokenizerUrl = `${baseUrl}/${tokenizerConfig.file}`;
          attemptedPaths.push(tokenizerUrl);
          try {
            const tokenizerText = await timedTokenizerLoadPhase(loadTiming, 'assetLoad', async () => {
              const response = await fetch(tokenizerUrl);
              if (!response.ok) {
                throw new Error(`Failed to fetch tokenizer: ${response.status}`);
              }
              return response.text();
            });
            tokenizerJson = timedTokenizerLoadPhaseSync(
              loadTiming,
              'assetParse',
              () => JSON.parse(tokenizerText)
            );
            loadTiming.assetSource = tokenizerUrl;
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            log.warn('Tokenizer', `Failed to fetch bundled tokenizer from URL: ${message}`);
          }
        } else if (!tokenizerJson) {
          // Try to load from OPFS (for cached models)
          attemptedPaths.push('OPFS:tokenizer.json');
          try {
            const tokenizerStr = await timedTokenizerLoadPhase(loadTiming, 'assetLoad', async () => {
              const { loadTokenizerFromStore } = await import('../storage/shard-manager.js');
              return loadTokenizerFromStore();
            });
            if (tokenizerStr) {
              tokenizerJson = timedTokenizerLoadPhaseSync(
                loadTiming,
                'assetParse',
                () => JSON.parse(tokenizerStr)
              );
              loadTiming.assetSource = 'OPFS:tokenizer.json';
            }
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            log.warn('Tokenizer', `Failed to load bundled tokenizer from OPFS: ${message}`);
          }
        }

        if (tokenizerJson) {
          timedTokenizerLoadPhaseSync(
            loadTiming,
            'backendLoad',
            () => (this.backend).load(tokenizerJson)
          );
          if (cacheKey) {
            timedTokenizerLoadPhaseSync(
              loadTiming,
              'cacheStore',
              () => BUNDLED_TOKENIZER_CACHE.set(cacheKey, this.backend)
            );
          }
          this.config = tokenizerConfig;
          finishTokenizerLoadTiming(loadTiming, 'complete', loadStart);
          return;
        }

        // No external fallback - bundled tokenizer is required
        throw new Error(
          `[Tokenizer] Bundled tokenizer not found for model "${modelId}". ` +
          `Expected tokenizer file: "${tokenizerConfig.file}". ` +
          `Attempted paths: [${attemptedPaths.join(', ')}]. ` +
          'Ensure tokenizer.json is in OPFS or model directory. ' +
          'Clear browser storage and re-download the model.'
        );
      }

      let hfModel = tokenizerConfig.hfModel ?? tokenizerConfig.modelId ?? null;
      const allowArchFallback = tokenizerConfig.allowArchFallback === true;
      if (allowArchFallback && !hfModel) {
        throw new Error(
          `[Tokenizer] tokenizer.allowArchFallback requires explicit tokenizer.hfModel or tokenizer.modelId for model "${modelId}".`
        );
      }

      if (hfModel) {
        // Use Transformers.js for HuggingFace models (fallback)
        loadTiming.backend = 'transformersjs';
        loadTiming.assetSource = hfModel;
        log.info('Tokenizer', `Loading from HuggingFace: ${hfModel}`);
        this.backend = timedTokenizerLoadPhaseSync(loadTiming, 'backendCreate', () => new TransformersTokenizer({
          modelId: hfModel,
          ...tokenizerConfig
        }));
        await timedTokenizerLoadPhase(loadTiming, 'backendLoad', () => (this.backend).load(hfModel));
      } else if (tokenizerConfig.sentencepieceModel) {
        // Load SentencePiece model
        loadTiming.backend = 'sentencepiece';
        loadTiming.tokenizerFile = typeof tokenizerConfig.sentencepieceModel === 'string'
          ? tokenizerConfig.sentencepieceModel
          : loadTiming.tokenizerFile;
        this.backend = timedTokenizerLoadPhaseSync(
          loadTiming,
          'backendCreate',
          () => new SentencePieceTokenizer(tokenizerConfig)
        );

        // Load the model data from the provided source

        let modelData;
        const spAttemptedPaths = [];
        if (typeof options.loadTokenizerModel === 'function') {
          spAttemptedPaths.push('custom-loader');
          try {
            modelData = await timedTokenizerLoadPhase(loadTiming, 'assetLoad', async () => {
              const loaded = await options.loadTokenizerModel(
                typeof tokenizerConfig.sentencepieceModel === 'string'
                  ? tokenizerConfig.sentencepieceModel
                  : undefined
              );
              return loaded == null
                ? null
                : toArrayBuffer(loaded, 'options.loadTokenizerModel');
            });
            if (modelData) {
              loadTiming.assetSource = 'custom-loader';
            }
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            log.warn('Tokenizer', `Failed to load sentencepiece model from custom loader: ${message}`);
          }
        }
        if (!modelData && tokenizerConfig.sentencepieceModel instanceof ArrayBuffer) {
          spAttemptedPaths.push('inline-ArrayBuffer');
          loadTiming.assetSource = 'inline-ArrayBuffer';
          modelData = tokenizerConfig.sentencepieceModel;
        } else if (!modelData && tokenizerConfig.loadShard) {
          // Use provided shard loader
          spAttemptedPaths.push('shard-loader');
          modelData = await timedTokenizerLoadPhase(
            loadTiming,
            'assetLoad',
            () => tokenizerConfig.loadShard(tokenizerConfig.sentencepieceModel)
          );
          loadTiming.assetSource = 'shard-loader';
        } else if (!modelData && typeof tokenizerConfig.sentencepieceModel === 'string') {
          if (options.baseUrl && !hasScheme(tokenizerConfig.sentencepieceModel)) {
            const url = `${options.baseUrl}/${tokenizerConfig.sentencepieceModel}`;
            spAttemptedPaths.push(url);
            modelData = await timedTokenizerLoadPhase(loadTiming, 'assetLoad', async () => {
              const response = await fetch(url);
              return response.arrayBuffer();
            });
            loadTiming.assetSource = url;
          } else if (hasScheme(tokenizerConfig.sentencepieceModel)) {
            spAttemptedPaths.push(tokenizerConfig.sentencepieceModel);
            modelData = await timedTokenizerLoadPhase(loadTiming, 'assetLoad', async () => {
              const response = await fetch(tokenizerConfig.sentencepieceModel);
              return response.arrayBuffer();
            });
            loadTiming.assetSource = tokenizerConfig.sentencepieceModel;
          } else {
            spAttemptedPaths.push('OPFS:tokenizer.model');
            try {
              modelData = await timedTokenizerLoadPhase(loadTiming, 'assetLoad', async () => {
                const { loadTokenizerModelFromStore } = await import('../storage/shard-manager.js');
                return loadTokenizerModelFromStore();
              });
              if (modelData) {
                loadTiming.assetSource = 'OPFS:tokenizer.model';
              }
            } catch (err) {
              const message = err instanceof Error ? err.message : String(err);
              log.warn('Tokenizer', `Failed to load tokenizer.model from OPFS: ${message}`);
            }
          }
        }

        if (modelData) {
          await timedTokenizerLoadPhase(loadTiming, 'backendLoad', () => (this.backend).load(modelData));
        } else {
          throw new Error(`Could not load SentencePiece model data. Attempted paths: [${spAttemptedPaths.join(', ')}]`);
        }
      } else if (tokenizerConfig.vocab && tokenizerConfig.merges) {
        // BPE with vocab + merges
        loadTiming.backend = 'bpe';
        this.backend = timedTokenizerLoadPhaseSync(loadTiming, 'backendCreate', () => new BPETokenizer(tokenizerConfig));
        timedTokenizerLoadPhaseSync(
          loadTiming,
          'backendLoad',
          () => (this.backend).load(tokenizerConfig.vocab, tokenizerConfig.merges)
        );
      } else {
        throw new Error(
          `[Tokenizer] No valid tokenizer configuration in manifest for model "${modelId}". ` +
          'Provide tokenizer.hfModel or bundle tokenizer.json (tokenizer.type="bundled", tokenizer.file="tokenizer.json").'
        );
      }

      this.config = tokenizerConfig;
      finishTokenizerLoadTiming(loadTiming, 'complete', loadStart);
    } catch (error) {
      finishTokenizerLoadTiming(loadTiming, 'failed', loadStart, error);
      throw error;
    }
  }
  encode(text) {
    if (!this.backend) {
      throw new Error('Tokenizer not initialized');
    }
    return this.backend.encode(text);
  }

  
  decode(ids, skipSpecialTokens = true, trim = true) {
    if (!this.backend) {
      throw new Error('Tokenizer not initialized');
    }
    return this.backend.decode(ids, skipSpecialTokens, trim);
  }

  
  getSpecialTokens() {
    return this.backend?.specialTokens || {};
  }

  
  getVocabSize() {
    return this.backend?.getVocabSize() || 0;
  }

  getHotTokenIds(limit) {
    if (!this.backend || typeof this.backend.getHotTokenIds !== 'function') {
      return null;
    }
    return this.backend.getHotTokenIds(limit);
  }

  getLoadTiming() {
    return cloneTokenizerLoadTiming(this.loadTiming);
  }
}

export default Tokenizer;
