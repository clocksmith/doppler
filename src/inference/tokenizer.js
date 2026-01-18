

import { log } from '../debug/index.js';
import { BaseTokenizer } from './tokenizers/base.js';
import { TransformersTokenizer, BundledTokenizer } from './tokenizers/bundled.js';
import { SentencePieceTokenizer } from './tokenizers/sentencepiece.js';
import { BPETokenizer } from './tokenizers/bpe.js';


export class Tokenizer {
  
  backend = null;

  
  config = null;

  
  async initialize(manifest, options = {}) {
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
        '[Tokenizer] Bundled tokenizer not found. ' +
        'Ensure tokenizer.json is in OPFS or model directory. ' +
        'Clear browser storage and re-download the model.'
      );
    }

    const hfModel = tokenizerConfig.hfModel;
    const allowArchFallback = tokenizerConfig.allowArchFallback === true;
    if (allowArchFallback && !hfModel) {
      throw new Error(
        '[Tokenizer] tokenizer.allowArchFallback is not supported. ' +
        'Provide tokenizer.hfModel or bundle tokenizer.json.'
      );
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
        '[Tokenizer] No valid tokenizer configuration in manifest. ' +
        'Provide tokenizer.hfModel or bundle tokenizer.json (tokenizer.type="bundled", tokenizer.file="tokenizer.json").'
      );
    }

    this.config = tokenizerConfig;
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
