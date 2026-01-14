/**
 * Tokenizer Wrapper
 *
 * Provides a unified interface for tokenization across different backends.
 *
 * @module inference/tokenizer
 */

import { log } from '../debug/index.js';
import { BaseTokenizer } from './tokenizers/base.js';
import { TransformersTokenizer, BundledTokenizer } from './tokenizers/bundled.js';
import { SentencePieceTokenizer } from './tokenizers/sentencepiece.js';
import { BPETokenizer } from './tokenizers/bpe.js';

/**
 * Tokenizer wrapper that auto-detects backend from model manifest
 * This is a thin wrapper over the backend implementations
 */
export class Tokenizer {
  /** @type {import('./tokenizers/base.js').BaseTokenizer | null} */
  backend = null;

  /** @type {import('./tokenizers/types.js').TokenizerConfig | null} */
  config = null;

  /**
   * Initialize from model manifest
   * @param {import('./tokenizers/types.js').ModelManifest} manifest
   * @param {{ baseUrl?: string }} [options]
   * @returns {Promise<void>}
   */
  async initialize(manifest, options = {}) {
    const tokenizerConfig = manifest.tokenizer || {};

    // Check for bundled or HuggingFace tokenizer first (eliminates transformers.js dependency)
    const isBundled = tokenizerConfig.type === 'bundled' || tokenizerConfig.type === 'huggingface';
    if (isBundled && tokenizerConfig.file) {
      log.info('Tokenizer', `Loading ${tokenizerConfig.type} tokenizer from ${tokenizerConfig.file}`);
      this.backend = new BundledTokenizer(tokenizerConfig);

      const baseUrl = options.baseUrl;
      /** @type {import('./tokenizers/types.js').HuggingFaceTokenizerJson | import('./tokenizers/types.js').BundledTokenizerJson | null} */
      let tokenizerJson = null;

      // Try to load tokenizer.json
      if (baseUrl) {
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
          const { loadTokenizerFromOPFS } = await import('../storage/shard-manager.js');
          const tokenizerStr = await loadTokenizerFromOPFS();
          if (tokenizerStr) {
            tokenizerJson = JSON.parse(tokenizerStr);
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          log.warn('Tokenizer', `Failed to load bundled tokenizer from OPFS: ${message}`);
        }
      }

      if (tokenizerJson) {
        /** @type {import('./tokenizers/bundled.js').BundledTokenizer} */ (this.backend).load(tokenizerJson);
        this.config = tokenizerConfig;
        return;
      }

      // No external fallback - bundled tokenizer is required
      throw new Error(
        '[Tokenizer] Bundled tokenizer not found. ' +
        'Ensure tokenizer.json is in OPFS or model directory. ' +
        'Clear browser storage and re-download the model.'
      );
    }

    // Try to infer HuggingFace model ID from manifest if not explicitly set
    let hfModel = tokenizerConfig.hfModel;
    if (!hfModel && !tokenizerConfig.sentencepieceModel && !tokenizerConfig.vocab) {
      hfModel = this._inferHuggingFaceModel(manifest);
    }

    if (hfModel) {
      // Use Transformers.js for HuggingFace models (fallback)
      log.info('Tokenizer', `Loading from HuggingFace: ${hfModel}`);
      this.backend = new TransformersTokenizer({
        modelId: hfModel,
        ...tokenizerConfig
      });
      await /** @type {import('./tokenizers/bundled.js').TransformersTokenizer} */ (this.backend).load(hfModel);
    } else if (tokenizerConfig.sentencepieceModel) {
      // Load SentencePiece model
      this.backend = new SentencePieceTokenizer(tokenizerConfig);

      // Load the model data from the provided source
      /** @type {ArrayBuffer | undefined} */
      let modelData;
      if (tokenizerConfig.sentencepieceModel instanceof ArrayBuffer) {
        modelData = tokenizerConfig.sentencepieceModel;
      } else if (tokenizerConfig.loadShard) {
        // Use provided shard loader
        modelData = await tokenizerConfig.loadShard(tokenizerConfig.sentencepieceModel);
      } else if (typeof tokenizerConfig.sentencepieceModel === 'string') {
        // Try to fetch as URL
        const response = await fetch(tokenizerConfig.sentencepieceModel);
        modelData = await response.arrayBuffer();
      }

      if (modelData) {
        await /** @type {import('./tokenizers/sentencepiece.js').SentencePieceTokenizer} */ (this.backend).load(modelData);
      } else {
        throw new Error('Could not load SentencePiece model data');
      }
    } else if (tokenizerConfig.vocab && tokenizerConfig.merges) {
      // BPE with vocab + merges
      this.backend = new BPETokenizer(tokenizerConfig);
      /** @type {import('./tokenizers/bpe.js').BPETokenizer} */ (this.backend).load(tokenizerConfig.vocab, tokenizerConfig.merges);
    } else {
      throw new Error('No valid tokenizer configuration in manifest');
    }

    this.config = tokenizerConfig;
  }

  /**
   * Infer HuggingFace model ID from manifest architecture
   * @param {import('./tokenizers/types.js').ModelManifest} manifest
   * @returns {string | null}
   */
  _inferHuggingFaceModel(manifest) {
    const arch = typeof manifest.architecture === 'string'
      ? manifest.architecture
      : (manifest.modelType || manifest.config?.architectures?.[0] || '');
    const archLower = arch.toLowerCase();

    // Map architecture names to public HuggingFace tokenizer repos
    // Using Xenova's repos where possible as they are optimized for Transformers.js
    /** @type {Record<string, string>} */
    const archToHF = {
      'gemma3': 'google/gemma-3-4b-it',    // Gemma 3 (fallback only - bundled tokenizer preferred)
      'gemma2': 'Xenova/gemma-tokenizer',
      'gemma': 'Xenova/gemma-tokenizer',
      'llama3': 'Xenova/llama3-tokenizer-new',
      'llama2': 'Xenova/llama2-tokenizer',
      'llama': 'Xenova/llama2-tokenizer',
      // Mistral v0.1/v0.2 has 32000 vocab, v0.3 has 32768 vocab with different tokenization
      'mistral': 'Xenova/mistral-tokenizer-v1',
      'mixtral': 'Xenova/mistral-tokenizer-v1',
      'qwen2': 'Xenova/qwen2.5-0.5b-instruct',
      'qwen': 'Xenova/qwen1.5-0.5b',
      'phi3': 'Xenova/phi-3-mini-4k-instruct',
      'phi': 'Xenova/phi-2',
      'smollm': 'HuggingFaceTB/SmolLM-360M-Instruct',
      'tinyllama': 'Xenova/TinyLlama-1.1B-Chat-v1.0',
    };

    // Check vocab size for Mistral - v0.3 has 32768 vocab with different tokenization
    const vocabSize = manifest.config?.vocab_size ||
                      manifest.config?.text_config?.vocab_size ||
                      manifest.tokenizer?.vocabSize;
    if ((archLower.includes('mistral') || archLower.includes('mixtral')) && vocabSize === 32768) {
      // Mistral v0.3+ with extended vocabulary needs the official tokenizer
      log.info('Tokenizer', 'Detected Mistral v0.3+ (vocab_size=32768), using official tokenizer');
      return 'mistralai/Mistral-7B-Instruct-v0.3';
    }

    for (const [key, hfModel] of Object.entries(archToHF)) {
      if (archLower.includes(key)) {
        log.info('Tokenizer', `Inferred HuggingFace model from architecture "${arch}": ${hfModel}`);
        return hfModel;
      }
    }

    // Check model type in config
    const modelType = manifest.config?.model_type || manifest.config?.text_config?.model_type || '';
    if (modelType) {
      for (const [key, hfModel] of Object.entries(archToHF)) {
        if (modelType.toLowerCase().includes(key)) {
          log.info('Tokenizer', `Inferred HuggingFace model from model_type "${modelType}": ${hfModel}`);
          return hfModel;
        }
      }
    }

    return null;
  }

  /**
   * Encode text to token IDs
   * @param {string} text
   * @returns {number[]}
   */
  encode(text) {
    if (!this.backend) {
      throw new Error('Tokenizer not initialized');
    }
    return this.backend.encode(text);
  }

  /**
   * Decode token IDs to text
   * @param {number[]} ids
   * @param {boolean} [skipSpecialTokens=true] - Whether to skip special tokens in output
   * @param {boolean} [trim=true] - Whether to trim whitespace (default true, set false for streaming)
   * @returns {string}
   */
  decode(ids, skipSpecialTokens = true, trim = true) {
    if (!this.backend) {
      throw new Error('Tokenizer not initialized');
    }
    return this.backend.decode(ids, skipSpecialTokens, trim);
  }

  /**
   * Get special tokens
   * @returns {import('./tokenizers/types.js').SpecialTokens}
   */
  getSpecialTokens() {
    return this.backend?.specialTokens || {};
  }

  /**
   * Get vocabulary size
   * @returns {number}
   */
  getVocabSize() {
    return this.backend?.getVocabSize() || 0;
  }
}

export default Tokenizer;
