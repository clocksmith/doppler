

import { log } from '../debug/index.js';
import { BaseTokenizer } from './tokenizers/base.js';
import { TransformersTokenizer, BundledTokenizer } from './tokenizers/bundled.js';
import { SentencePieceTokenizer } from './tokenizers/sentencepiece.js';
import { BPETokenizer } from './tokenizers/bpe.js';

function hasScheme(value) {
  return /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(value);
}

function toArrayBuffer(value, label) {
  if (value instanceof ArrayBuffer) {
    return value;
  }
  if (value instanceof Uint8Array) {
    return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  }
  throw new Error(`${label} must return ArrayBuffer or Uint8Array.`);
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

export class Tokenizer {
  
  backend = null;

  
  config = null;

  
  async initialize(manifest, options = {}) {
    const modelId = manifest?.modelId || 'unknown';
    // Merge tokenizer config: preset provides fallback hints, manifest takes precedence
    const presetTokenizer = options.presetTokenizer || {};
    const tokenizerConfig = { ...presetTokenizer, ...(manifest.tokenizer || {}) };
    const eosTokenId = Array.isArray(manifest.eos_token_id)
      ? manifest.eos_token_id[0]
      : manifest.eos_token_id;

    if (tokenizerConfig.eosToken == null && Array.isArray(tokenizerConfig.eosTokens) && tokenizerConfig.eosTokens.length > 0) {
      tokenizerConfig.eosToken = tokenizerConfig.eosTokens[0];
    }
    if (tokenizerConfig.eosToken == null && typeof eosTokenId === 'number') {
      tokenizerConfig.eosToken = eosTokenId;
    }
    if (tokenizerConfig.bosToken == null && tokenizerConfig.bosTokenId != null) {
      tokenizerConfig.bosToken = tokenizerConfig.bosTokenId;
    }
    if (tokenizerConfig.padToken == null && tokenizerConfig.padTokenId != null) {
      tokenizerConfig.padToken = tokenizerConfig.padTokenId;
    }
    if (tokenizerConfig.unkToken == null && tokenizerConfig.unkTokenId != null) {
      tokenizerConfig.unkToken = tokenizerConfig.unkTokenId;
    }

    // Check for bundled or HuggingFace tokenizer first (eliminates transformers.js dependency)
    const isBundled = tokenizerConfig.type === 'bundled' || tokenizerConfig.type === 'huggingface';
    if (isBundled && tokenizerConfig.file) {
      tokenizerConfig.deferSpecialTokens = true;
      log.info('Tokenizer', `Loading ${tokenizerConfig.type} tokenizer from ${tokenizerConfig.file}`);
      this.backend = new BundledTokenizer(tokenizerConfig);

      const baseUrl = options.baseUrl;
      
      let tokenizerJson = null;

      // Try to load tokenizer.json
      if (typeof options.loadTokenizerJson === 'function') {
        try {
          const loaded = await options.loadTokenizerJson();
          tokenizerJson = parseTokenizerJsonPayload(loaded);
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          log.warn('Tokenizer', `Failed to load bundled tokenizer from custom loader: ${message}`);
        }
      }

      if (!tokenizerJson && baseUrl) {
        // Load from remote URL
        const tokenizerUrl = `${baseUrl}/${tokenizerConfig.file}`;
        try {
          const response = await fetch(tokenizerUrl);
          if (!response.ok) {
            throw new Error(`Failed to fetch tokenizer: ${response.status}`);
          }
          tokenizerJson = await response.json();
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          log.warn('Tokenizer', `Failed to fetch bundled tokenizer from URL: ${message}`);
        }
      } else {
        // Try to load from OPFS (for cached models)
        try {
          const { loadTokenizerFromStore } = await import('../storage/shard-manager.js');
          const tokenizerStr = await loadTokenizerFromStore();
          if (tokenizerStr) {
            tokenizerJson = JSON.parse(tokenizerStr);
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          log.warn('Tokenizer', `Failed to load bundled tokenizer from OPFS: ${message}`);
        }
      }

      if (tokenizerJson) {
         (this.backend).load(tokenizerJson);
        this.config = tokenizerConfig;
        return;
      }

      // No external fallback - bundled tokenizer is required
      throw new Error(
        `[Tokenizer] Bundled tokenizer not found for model "${modelId}". ` +
        `Expected tokenizer file: "${tokenizerConfig.file}". ` +
        'Ensure tokenizer.json is in OPFS or model directory. ' +
        'Clear browser storage and re-download the model.'
      );
    }

    let hfModel = tokenizerConfig.hfModel;
    const allowArchFallback = tokenizerConfig.allowArchFallback === true;
    if (allowArchFallback && !hfModel) {
      const inferred = this._inferHuggingFaceModel(manifest);
      if (inferred) {
        hfModel = inferred;
        log.warn('Tokenizer', `Using inferred HuggingFace model: ${inferred}`);
      }
    }

    if (hfModel) {
      // Use Transformers.js for HuggingFace models (fallback)
      log.info('Tokenizer', `Loading from HuggingFace: ${hfModel}`);
      this.backend = new TransformersTokenizer({
        modelId: hfModel,
        ...tokenizerConfig
      });
      await  (this.backend).load(hfModel);
    } else if (tokenizerConfig.sentencepieceModel) {
      // Load SentencePiece model
      this.backend = new SentencePieceTokenizer(tokenizerConfig);

      // Load the model data from the provided source
      
      let modelData;
      if (typeof options.loadTokenizerModel === 'function') {
        try {
          const loaded = await options.loadTokenizerModel(
            typeof tokenizerConfig.sentencepieceModel === 'string'
              ? tokenizerConfig.sentencepieceModel
              : undefined
          );
          if (loaded != null) {
            modelData = toArrayBuffer(loaded, 'options.loadTokenizerModel');
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          log.warn('Tokenizer', `Failed to load sentencepiece model from custom loader: ${message}`);
        }
      }
      if (!modelData && tokenizerConfig.sentencepieceModel instanceof ArrayBuffer) {
        modelData = tokenizerConfig.sentencepieceModel;
      } else if (!modelData && tokenizerConfig.loadShard) {
        // Use provided shard loader
        modelData = await tokenizerConfig.loadShard(tokenizerConfig.sentencepieceModel);
      } else if (!modelData && typeof tokenizerConfig.sentencepieceModel === 'string') {
        if (options.baseUrl && !hasScheme(tokenizerConfig.sentencepieceModel)) {
          const url = `${options.baseUrl}/${tokenizerConfig.sentencepieceModel}`;
          const response = await fetch(url);
          modelData = await response.arrayBuffer();
        } else if (hasScheme(tokenizerConfig.sentencepieceModel)) {
          const response = await fetch(tokenizerConfig.sentencepieceModel);
          modelData = await response.arrayBuffer();
        } else {
          try {
            const { loadTokenizerModelFromStore } = await import('../storage/shard-manager.js');
            modelData = await loadTokenizerModelFromStore();
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            log.warn('Tokenizer', `Failed to load tokenizer.model from OPFS: ${message}`);
          }
        }
      }

      if (modelData) {
        await  (this.backend).load(modelData);
      } else {
        throw new Error('Could not load SentencePiece model data');
      }
    } else if (tokenizerConfig.vocab && tokenizerConfig.merges) {
      // BPE with vocab + merges
      this.backend = new BPETokenizer(tokenizerConfig);
       (this.backend).load(tokenizerConfig.vocab, tokenizerConfig.merges);
    } else {
      throw new Error(
        `[Tokenizer] No valid tokenizer configuration in manifest for model "${modelId}". ` +
        'Provide tokenizer.hfModel or bundle tokenizer.json (tokenizer.type="bundled", tokenizer.file="tokenizer.json").'
      );
    }

    this.config = tokenizerConfig;
  }

  
  _inferHuggingFaceModel(manifest) {
    const tokenizer = manifest?.tokenizer ?? {};
    if (typeof tokenizer.modelId === 'string' && tokenizer.modelId.length > 0) {
      return tokenizer.modelId;
    }
    if (typeof tokenizer.hfModel === 'string' && tokenizer.hfModel.length > 0) {
      return tokenizer.hfModel;
    }
    if (typeof manifest?.modelId === 'string' && manifest.modelId.length > 0) {
      return manifest.modelId;
    }
    return null;
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
}

export default Tokenizer;
