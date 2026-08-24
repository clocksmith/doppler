import { BaseTokenizer } from './base.js';
import { log } from '../../debug/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import { inferBundledTokenizerBehaviorFlags } from './behavior-flags.js';
import {
  createByteLevelCodec,
  decodeByteFallbackTokens,
  decodeByteLevelTokens,
  encodeByteLevelText,
  resolveByteLevelPretokenizerConfig,
} from './bundled/decoders.js';
import {
  appendVocabEntry,
  buildMergeRanks,
  loadObjectVocab,
  rankFallbackBpeHotTokenIds,
  registerAddedTokens,
  resolveSpecialTokens,
} from './bundled/registry.js';

function normalizeInlineCaseInsensitiveGroups(pattern) {
  let source = String(pattern || '');
  let caseInsensitive = false;
  let start = source.indexOf('(?i:');
  while (start !== -1) {
    let depth = 1;
    let escaped = false;
    let end = start + 4;
    for (; end < source.length; end++) {
      const char = source[end];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === '\\') {
        escaped = true;
        continue;
      }
      if (char === '(') depth += 1;
      else if (char === ')') {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    if (depth !== 0) {
      throw new Error('[Tokenizer] Unterminated inline case-insensitive regex group.');
    }
    source = `${source.slice(0, start)}(?:${source.slice(start + 4, end)})${source.slice(end + 1)}`;
    caseInsensitive = true;
    start = source.indexOf('(?i:', start + 3);
  }
  return { source, caseInsensitive };
}

function resolveIsolatedSplitPretokenizer(preTokenizer) {
  if (!preTokenizer || typeof preTokenizer !== 'object') return null;
  if (preTokenizer.type === 'Sequence' && Array.isArray(preTokenizer.pretokenizers)) {
    for (const entry of preTokenizer.pretokenizers) {
      const resolved = resolveIsolatedSplitPretokenizer(entry);
      if (resolved) return resolved;
    }
    return null;
  }
  if (
    preTokenizer.type !== 'Split'
    || preTokenizer.behavior !== 'Isolated'
    || preTokenizer.invert === true
    || typeof preTokenizer.pattern?.Regex !== 'string'
  ) {
    return null;
  }
  const normalized = normalizeInlineCaseInsensitiveGroups(preTokenizer.pattern.Regex);
  const flags = `gu${normalized.caseInsensitive ? 'i' : ''}`;
  try {
    new RegExp(normalized.source, flags);
  } catch (error) {
    throw new Error(`[Tokenizer] Unsupported split pre-tokenizer regex: ${error.message}`);
  }
  return { source: normalized.source, flags };
}

function resolvesCharacterSplitPretokenizer(preTokenizer) {
  if (!preTokenizer || typeof preTokenizer !== 'object') return false;
  if (preTokenizer.type === 'Sequence' && Array.isArray(preTokenizer.pretokenizers)) {
    return preTokenizer.pretokenizers.some(resolvesCharacterSplitPretokenizer);
  }
  return preTokenizer.type === 'Split'
    && preTokenizer.behavior === 'Removed'
    && preTokenizer.invert !== true
    && preTokenizer.pattern?.String === '';
}

export class TransformersTokenizer extends BaseTokenizer {
  #tokenizer = null;
  #modelId;
  constructor(config = {}) {
    // TransformersTokenizer gets vocabSize from setTokenizer(), so defer validation
    super({
      ...config,
    });
    this.#modelId = config.modelId;
  }

  
  setTokenizer(tokenizer) {
    this.#tokenizer = tokenizer;
    if (tokenizer.model?.vocab) {
      this.vocabSize = Object.keys(tokenizer.model.vocab).length;
    }
  }

  
  async load(_modelId) {
    // DOPPLER uses bundled tokenizers only - no external CDN dependencies
    throw new Error(
      '[Tokenizer] TransformersTokenizer is deprecated. ' +
      'Use bundled tokenizer (type: "bundled" or "huggingface" with file). ' +
      'DOPPLER requires no external runtime dependencies.'
    );
  }

  
  encode(text) {
    if (!this.#tokenizer) {
      throw new Error('Tokenizer not initialized');
    }

    const result = this.#tokenizer.encode(text, {
      add_special_tokens: this.addBosToken
    });

    return Array.from(result);
  }

  
  decode(ids, skipSpecialTokens = true, trim = true) {
    if (!this.#tokenizer) {
      throw new Error('Tokenizer not initialized');
    }

    const result = this.#tokenizer.decode(ids, { skip_special_tokens: skipSpecialTokens });
    return trim ? result.trim() : result;
  }

  
  batchEncode(texts) {
    return texts.map(t => this.encode(t));
  }

  getHotTokenIds(limit) {
    void limit;
    return null;
  }
}


export class BundledTokenizer extends BaseTokenizer {
  
  #vocab = new Map();
  
  #reverseVocab = new Map();
  
  #merges = [];
  
  #mergeRanks = new Map();
  
  #scores = [];
  
  #tokenTypes = [];
  
  #type = null;
  
  #byteTokens = new Map();
  
  #specialTokenPatterns = [];
  
  #specialTokenIds = new Set();
  
  #addSpacePrefix = true;
  
  #spacePrefixChar = '▁';
  
  #byteDecoder = null;

  #byteEncoder = null;

  #useByteLevelEncoding = false;

  #splitPretokenizer = null;

  #splitEveryCharacter = false;

  #wordPiecePrefix = '##';

  #wordPieceMaxChars = 100;

  
  constructor(config = {}) {
    // BundledTokenizer gets vocabSize from load(), so defer validation
    super({
      ...config,
    });
  }

  #resetState() {
    this.#vocab.clear();
    this.#reverseVocab.clear();
    this.#merges = [];
    this.#mergeRanks.clear();
    this.#scores = [];
    this.#tokenTypes = [];
    this.#type = null;
    this.#byteTokens.clear();
    this.#specialTokenPatterns = [];
    this.#specialTokenIds = new Set();
    this.#addSpacePrefix = true;
    this.#spacePrefixChar = '▁';
    this.#byteDecoder = null;
    this.#byteEncoder = null;
    this.#useByteLevelEncoding = false;
    this.#splitPretokenizer = null;
    this.#splitEveryCharacter = false;
    this.#wordPiecePrefix = '##';
    this.#wordPieceMaxChars = 100;
    this.vocabSize = 0;
  }

  
  isSpecialToken(tokenId) {
    if (this.#specialTokenIds.size > 0) {
      return this.#specialTokenIds.has(tokenId);
    }
    return super.isSpecialToken(tokenId);
  }

  
  #getUnkTokenId() {
    if (this.specialTokens.unk == null) {
      throw new Error('[Tokenizer] Missing unk token in tokenizer metadata.');
    }
    return this.specialTokens.unk;
  }

  load(tokenizerJson) {
    this.#resetState();
    // Detect format: HuggingFace has model.vocab, bundled has top-level vocab
    const isHuggingFace = 'model' in tokenizerJson && tokenizerJson.model?.vocab !== undefined;

    if (isHuggingFace) {
      this.#loadHuggingFaceFormat( (tokenizerJson));
    } else {
      this.#loadBundledFormat( (tokenizerJson));
    }
  }

  
  #loadHuggingFaceFormat(hf) {
    const model = hf.model;
    if (typeof model.type !== 'string') {
      throw new Error('[Tokenizer] Missing model.type in HuggingFace tokenizer JSON.');
    }
    this.#type = model.type.toLowerCase();
    if (this.#type !== 'bpe' && this.#type !== 'unigram' && this.#type !== 'wordpiece') {
      throw new Error(`[Tokenizer] Unsupported tokenizer type: ${model.type}`);
    }
    log.info('Tokenizer', `HuggingFace model.type="${model.type}", using type="${this.#type}"`);
    this.#byteDecoder = null;
    let maxId = -1;

    // Handle vocab based on type
    if (this.#type === 'unigram' && Array.isArray(model.vocab)) {
      // Unigram format: [[token, score], ...]
      for (let i = 0; i < model.vocab.length; i++) {
        const [token, score] = model.vocab[i];
        appendVocabEntry(token, i, this.#vocab, this.#reverseVocab, this.#byteTokens);
        this.#scores.push(score);
        if (i > maxId) maxId = i;
      }
    } else if (
      (this.#type === 'bpe' || this.#type === 'wordpiece')
      && model.vocab
      && typeof model.vocab === 'object'
    ) {
      // BPE and WordPiece format: { token: id }
      maxId = loadObjectVocab(model.vocab, this.#vocab, this.#reverseVocab, this.#byteTokens);
    } else {
      throw new Error(`[Tokenizer] Missing vocab for tokenizer type: ${model.type}`);
    }

    this.vocabSize = this.#vocab.size;

    // Load merges from model.merges
    // Handle both string format ("token1 token2") and array format (["token1", "token2"])
    if (model.merges && model.merges.length > 0) {
      this.#merges = buildMergeRanks(model.merges, this.#mergeRanks);
    }

    const addedTokens = Array.isArray(hf.added_tokens) ? hf.added_tokens : [];
    const specialTokenIds = new Set();
    const specialTokenPatterns = [];
    const derivedSpecialTokens = {
      pad: null,
      bos: null,
      eos: null,
      unk: null,
    };
    const addedMaxId = registerAddedTokens(
      addedTokens,
      this.#vocab,
      this.#reverseVocab,
      specialTokenPatterns,
      specialTokenIds,
      derivedSpecialTokens
    );
    if (addedMaxId > maxId) {
      maxId = addedMaxId;
    }

    const specialTokensRaw = hf.special_tokens_map || hf.specialTokens || hf.special_tokens || null;
    const fallbackTokens = {
      ...this.specialTokens,
      pad: model.pad_id ?? this.specialTokens.pad ?? derivedSpecialTokens.pad,
      bos: model.bos_id ?? this.specialTokens.bos ?? derivedSpecialTokens.bos,
      eos: model.eos_id ?? this.specialTokens.eos ?? derivedSpecialTokens.eos,
      unk: model.unk_id ?? this.specialTokens.unk ?? derivedSpecialTokens.unk,
    };
    this.specialTokens = resolveSpecialTokens(specialTokensRaw, fallbackTokens, this.#vocab, {
      allowMissingEos: hf.add_eos_token === false || hf.addEosToken === false,
    });
    this.#specialTokenIds = specialTokenIds;
    this.#specialTokenPatterns = specialTokenPatterns;
    const builtinSpecials = [
      this.specialTokens.pad,
      this.specialTokens.bos,
      this.specialTokens.eos,
      this.specialTokens.unk,
    ];
    for (const id of builtinSpecials) {
      if (typeof id === 'number' && Number.isFinite(id)) {
        this.#specialTokenIds.add(id);
      }
    }
    // Sort special tokens by length (longest first) for greedy matching
    this.#specialTokenPatterns.sort((a, b) => b.content.length - a.content.length);
    // Debug: log special tokens
    log.debug('Tokenizer', `Special token patterns: ${this.#specialTokenPatterns.map(t => `${t.id}:"${t.content}"`).join(', ')}`);

    // Some models add special tokens with IDs above the base vocab range.
    // Keep vocabSize aligned to the maximum ID + 1 to match embedding/LM-head shapes.
    if (maxId >= 0) {
      this.vocabSize = Math.max(this.vocabSize, maxId + 1);
    }

    // Handle behavior flags (use HF config if present, else runtime defaults)
    const runtimeDefaults = getRuntimeConfig().inference.tokenizer;
    const byteLevelPretokenizer = resolveByteLevelPretokenizerConfig(hf.pre_tokenizer);
    this.#splitPretokenizer = resolveIsolatedSplitPretokenizer(hf.pre_tokenizer);
    this.#splitEveryCharacter = resolvesCharacterSplitPretokenizer(hf.pre_tokenizer);
    if (this.#type === 'wordpiece') {
      this.#wordPiecePrefix = typeof model.continuing_subword_prefix === 'string'
        ? model.continuing_subword_prefix
        : '##';
      this.#wordPieceMaxChars = Number.isInteger(model.max_input_chars_per_word)
        && model.max_input_chars_per_word > 0
        ? model.max_input_chars_per_word
        : 100;
    }
    const configuredAddBosToken = this.addBosToken;
    const configuredAddEosToken = this.addEosToken;
    const inferredFlags = inferBundledTokenizerBehaviorFlags(hf, this.specialTokens);
    this.addBosToken =
      hf.add_bos_token
      ?? hf.addBosToken
      ?? configuredAddBosToken
      ?? inferredFlags.addBosToken
      ?? runtimeDefaults.addBosToken;
    this.addEosToken =
      hf.add_eos_token
      ?? hf.addEosToken
      ?? configuredAddEosToken
      ?? inferredFlags.addEosToken
      ?? runtimeDefaults.addEosToken;
    if (this.addBosToken && this.specialTokens.bos == null) {
      throw new Error('[Tokenizer] addBosToken is enabled but bos token is missing.');
    }
    if (this.addEosToken && this.specialTokens.eos == null) {
      throw new Error('[Tokenizer] addEosToken is enabled but eos token is missing.');
    }
    // Determine if we should add a space prefix to the input
    // Check multiple locations where HuggingFace stores this:
    // - model.add_prefix_space / model.add_dummy_prefix (GPT-style)
    // - decoder.add_prefix_space (SentencePiece decoder)
    // - decoder.prepend_scheme === "always" (Metaspace decoder, used by Gemma)
    // - normalizer.prepend_scheme === "always" (Metaspace normalizer)
    // - runtime config addSpacePrefix (user override or null for auto-detect)
    const decoderPrepend = hf.decoder?.prepend_scheme === 'always' || hf.decoder?.add_prefix_space === true;
    const normalizerPrepend = hf.normalizer?.prepend_scheme === 'always' || hf.normalizer?.add_prefix_space === true;
    this.#useByteLevelEncoding = byteLevelPretokenizer.useByteLevel;
    if (this.#type === 'bpe' && this.#useByteLevelEncoding) {
      const codec = createByteLevelCodec();
      this.#byteDecoder = codec.decoder;
      this.#byteEncoder = codec.encoder;
    }
    const runtimeSpacePrefix = runtimeDefaults.addSpacePrefix;
    // Use explicit runtime config if set (non-null), otherwise auto-detect from tokenizer.json
    this.#addSpacePrefix = runtimeSpacePrefix
      ?? byteLevelPretokenizer.addPrefixSpace
      ?? model.add_prefix_space
      ?? model.add_dummy_prefix
      ?? decoderPrepend
      ?? normalizerPrepend
      ?? false;
    log.debug('Tokenizer', `addSpacePrefix=${this.#addSpacePrefix} (runtime=${runtimeSpacePrefix}, model=${model.add_prefix_space ?? model.add_dummy_prefix}, decoder=${decoderPrepend}, normalizer=${normalizerPrepend})`);

    // Detect space prefix style by checking which WORD tokens exist in vocab
    // GPT-style uses 'Ġ' (U+0120), SentencePiece uses '▁' (U+2581)
    // IMPORTANT: Only check for actual word tokens like 'Ġthe', not single char 'Ġ'
    // because some models (Gemma) have 'Ġ' as a token but use '▁' for actual word prefixes
    const hasGptStyle = this.#vocab.has('Ġthe') || this.#vocab.has('Ġa') || this.#vocab.has('Ġis');
    const hasSentencePieceStyle = this.#vocab.has('▁the') || this.#vocab.has('▁a') || this.#vocab.has('▁is');
    if (hasGptStyle && !hasSentencePieceStyle) {
      this.#spacePrefixChar = 'Ġ';
      log.debug('Tokenizer', 'Detected GPT-style space prefix');
    } else if (hasSentencePieceStyle && !hasGptStyle) {
      this.#spacePrefixChar = '▁';
      log.debug('Tokenizer', 'Detected SentencePiece-style space prefix');
    } else if (hasGptStyle && hasSentencePieceStyle) {
      // Both styles exist - prefer GPT style for Llama-family models
      this.#spacePrefixChar = 'Ġ';
      log.debug('Tokenizer', 'Both space styles found, defaulting to GPT-style');
    } else {
      // Neither style found - might be byte-level or no space prefix needed
      this.#addSpacePrefix = false;
      log.debug('Tokenizer', 'No space prefix tokens found, disabling space prefix');
    }

    log.info('Tokenizer', `Loaded HuggingFace ${this.vocabSize} tokens (${this.#type}), ${this.#specialTokenPatterns.length} special patterns, ${this.#merges.length} merges`);
    // Debug: show sample vocab entries (look for common words)
    const commonWords = ['the', '▁the', 'Ġthe', 'a', '▁a', 'is', '▁is', 'user', '▁user', 'u', 's', 'e', 'r'];
    const foundTokens = commonWords.map(w => {
      const id = this.#vocab.get(w);
      return id !== undefined ? `"${w}"=${id}` : null;
    }).filter(Boolean);
    log.debug('Tokenizer', `Common tokens in vocab: ${foundTokens.join(', ') || 'NONE FOUND'}`);
    // Show first few merges (escape whitespace)
    if (this.#merges.length > 0) {
      const escapedMerges = this.#merges.slice(0, 5).map(m =>
        String(m).replace(/\n/g, '\\n').replace(/\r/g, '\\r').replace(/ /g, '␣')
      );
      log.debug('Tokenizer', `First 5 merges: ${escapedMerges.join(' | ')}`);
    }
  }

  
  #loadBundledFormat(tokenizerJson) {
    if (typeof tokenizerJson.type !== 'string') {
      throw new Error('[Tokenizer] Missing tokenizer.type in bundled tokenizer JSON.');
    }
    this.#type = tokenizerJson.type.toLowerCase();
    if (this.#type !== 'bpe' && this.#type !== 'unigram') {
      throw new Error(`[Tokenizer] Unsupported tokenizer type: ${tokenizerJson.type}`);
    }
    this.#byteDecoder = null;

    // Build vocab maps
    let maxId = loadObjectVocab(tokenizerJson.vocab, this.#vocab, this.#reverseVocab, this.#byteTokens);

    this.vocabSize = this.#vocab.size;

    // Load merges for BPE
    if (tokenizerJson.merges && tokenizerJson.merges.length > 0) {
      this.#merges = buildMergeRanks(tokenizerJson.merges, this.#mergeRanks);
    }

    // Load scores for Unigram
    if (tokenizerJson.scores && tokenizerJson.scores.length > 0) {
      this.#scores = tokenizerJson.scores;
    }

    // Load token types if available
    if (tokenizerJson.tokenTypes) {
      this.#tokenTypes = tokenizerJson.tokenTypes;
    }

    const addedTokens = Array.isArray(tokenizerJson.added_tokens) ? tokenizerJson.added_tokens : [];
    const tokenPatterns = [];
    const specialTokenIds = new Set();
    const derivedSpecialTokens = {
      pad: null,
      bos: null,
      eos: null,
      unk: null,
    };
    const addedMaxId = registerAddedTokens(
      addedTokens,
      this.#vocab,
      this.#reverseVocab,
      tokenPatterns,
      specialTokenIds,
      derivedSpecialTokens
    );
    if (addedMaxId > maxId) {
      maxId = addedMaxId;
    }

    // Set special tokens - support both camelCase and snake_case formats
    const specialTokensRaw =  (tokenizerJson.specialTokens ||  (tokenizerJson).special_tokens);
    this.specialTokens = resolveSpecialTokens(
      specialTokensRaw,
      {
        ...derivedSpecialTokens,
        ...this.specialTokens,
      },
      this.#vocab,
      {
        allowMissingEos: tokenizerJson.add_eos_token === false || tokenizerJson.addEosToken === false,
      }
    );
    log.debug('Tokenizer', `Special tokens: BOS=${this.specialTokens.bos}, EOS=${this.specialTokens.eos}`);
    this.#specialTokenIds = specialTokenIds;
    this.#specialTokenPatterns = tokenPatterns;
    const builtinSpecials = [
      this.specialTokens.pad,
      this.specialTokens.bos,
      this.specialTokens.eos,
      this.specialTokens.unk,
    ];
    for (const id of builtinSpecials) {
      if (typeof id === 'number' && Number.isFinite(id)) {
        this.#specialTokenIds.add(id);
      }
    }
    this.#specialTokenPatterns.sort((a, b) => b.content.length - a.content.length);
    if (maxId >= 0) {
      this.vocabSize = Math.max(this.vocabSize, maxId + 1);
    }

    const runtimeDefaults = getRuntimeConfig().inference.tokenizer;
    const byteLevelPretokenizer = resolveByteLevelPretokenizerConfig(tokenizerJson.pre_tokenizer);
    this.#splitPretokenizer = resolveIsolatedSplitPretokenizer(tokenizerJson.pre_tokenizer);
    const configuredAddBosToken = this.addBosToken;
    const configuredAddEosToken = this.addEosToken;
    const inferredFlags = inferBundledTokenizerBehaviorFlags(tokenizerJson, this.specialTokens);
    this.addBosToken =
      tokenizerJson.addBosToken
      ?? tokenizerJson.add_bos_token
      ?? configuredAddBosToken
      ?? inferredFlags.addBosToken
      ?? runtimeDefaults.addBosToken;
    this.addEosToken =
      tokenizerJson.addEosToken
      ?? tokenizerJson.add_eos_token
      ?? configuredAddEosToken
      ?? inferredFlags.addEosToken
      ?? runtimeDefaults.addEosToken;
    if (this.addBosToken && this.specialTokens.bos == null) {
      throw new Error('[Tokenizer] addBosToken is enabled but bos token is missing.');
    }
    if (this.addEosToken && this.specialTokens.eos == null) {
      throw new Error('[Tokenizer] addEosToken is enabled but eos token is missing.');
    }
    this.#useByteLevelEncoding = byteLevelPretokenizer.useByteLevel;
    if (this.#type === 'bpe' && this.#useByteLevelEncoding) {
      const codec = createByteLevelCodec();
      this.#byteDecoder = codec.decoder;
      this.#byteEncoder = codec.encoder;
    }
    // NOTE: Default to FALSE - first word shouldn't get space prefix
    // Space prefixes are only for words that follow a space in original text
    this.#addSpacePrefix = tokenizerJson.addSpacePrefix === true
      || byteLevelPretokenizer.addPrefixSpace === true;

    // Detect space prefix style based on vocab tokens
    // GPT-style uses 'Ġ' (U+0120), SentencePiece uses '▁' (U+2581)
    // IMPORTANT: Only check for actual word tokens like 'Ġthe', not single char 'Ġ'
    // because some models (Gemma) have 'Ġ' as a token but use '▁' for actual word prefixes
    const hasGptStyle = this.#vocab.has('Ġthe') || this.#vocab.has('Ġa') || this.#vocab.has('Ġis');
    const hasSentencePieceStyle = this.#vocab.has('▁the') || this.#vocab.has('▁a') || this.#vocab.has('▁is');

    if (hasGptStyle && !hasSentencePieceStyle) {
      this.#spacePrefixChar = 'Ġ';
      log.debug('Tokenizer', 'Detected GPT-style space prefix');
    } else if (hasSentencePieceStyle && !hasGptStyle) {
      this.#spacePrefixChar = '▁';
      log.debug('Tokenizer', 'Detected SentencePiece-style space prefix');
    } else if (hasGptStyle && hasSentencePieceStyle) {
      // Both exist - prefer GPT-style for Llama 3 compatibility
      this.#spacePrefixChar = 'Ġ';
      log.debug('Tokenizer', 'Both space prefix styles found, using GPT-style');
    } else {
      // Default to SentencePiece style
      this.#spacePrefixChar = '▁';
      log.debug('Tokenizer', 'No space prefix tokens found, defaulting to SentencePiece-style');
    }

    log.info('Tokenizer', `Loaded ${this.vocabSize} tokens (${this.#type})`);
  }

  
  encode(text) {
    if (this.#vocab.size === 0) {
      throw new Error('BundledTokenizer not loaded');
    }

    
    const ids = [];

    if (this.addBosToken) {
      ids.push(this.specialTokens.bos);
    }

    // Split text around literal added tokens and special tokens, then tokenize
    // the remaining plain-text segments normally.
    const segments = this.#splitOnSpecialTokens(text);
    for (const seg of segments) {
      if (seg.isSpecial && seg.id !== undefined) {
        ids.push(seg.id);
      } else if (seg.text && seg.text.length > 0) {
        if (this.#type === 'unigram') {
          ids.push(...this.#encodeUnigram(seg.text));
        } else if (this.#type === 'wordpiece') {
          ids.push(...this.#encodeWordPiece(seg.text));
        } else {
          ids.push(...this.#encodeBPE(seg.text));
        }
      }
    }

    if (this.addEosToken) {
      ids.push(this.specialTokens.eos);
    }

    return ids;
  }

  
  #splitOnSpecialTokens(text) {
    if (this.#specialTokenPatterns.length === 0) {
      return [{ text, isSpecial: false }];
    }

    
    const segments = [];
    let remaining = text;

    while (remaining.length > 0) {
      // Find the EARLIEST special token match
      let earliestIdx = Infinity;
      
      let earliestToken = null;

      for (const { content, id } of this.#specialTokenPatterns) {
        const idx = remaining.indexOf(content);
        if (idx !== -1 && idx < earliestIdx) {
          earliestIdx = idx;
          earliestToken = { content, id };
        }
      }

      if (earliestToken === null) {
        // No special tokens found, rest is plain text
        segments.push({ text: remaining, isSpecial: false });
        break;
      }

      if (earliestIdx === 0) {
        // Special token at start
        segments.push({ id: earliestToken.id, isSpecial: true });
        remaining = remaining.slice(earliestToken.content.length);
      } else {
        // Text before special token
        segments.push({ text: remaining.slice(0, earliestIdx), isSpecial: false });
        segments.push({ id: earliestToken.id, isSpecial: true });
        remaining = remaining.slice(earliestIdx + earliestToken.content.length);
      }
    }

    return segments;
  }

  
  #encodeUnigram(text) {
    if (text.length === 0) return [];

    // Add space prefix at start if configured (SentencePiece add_dummy_prefix)
    // This matches HuggingFace behavior: "The" -> " The" -> "▁The"
    let normalized = text;
    if (this.#addSpacePrefix && !normalized.startsWith(' ')) {
      normalized = ` ${normalized}`;
    }

    // Normalize: convert spaces to the model's space prefix character
    // This turns " The color of" into "▁The▁color▁of"
    const sp = this.#spacePrefixChar;
    const prefixed = normalized.replace(/ /g, sp);

    const n = prefixed.length;
    if (n === 0) return [];

    // Viterbi: best[i] = {score, prev, tokenLen} for position i
    
    const best = new Array(n + 1).fill(null);
    best[0] = { score: 0, prev: -1, tokenLen: 0 };

    for (let i = 0; i < n; i++) {
      if (best[i] === null) continue;

      // Try all possible tokens starting at position i
      for (let len = 1; len <= Math.min(n - i, 32); len++) {
        const substr = prefixed.slice(i, i + len);
        const tokenId = this.#vocab.get(substr);

        if (tokenId !== undefined) {
          const score = this.#scores[tokenId] || 0;
          const newScore = best[i].score + score;
          if (best[i + len] === null || newScore > best[i + len].score) {
            best[i + len] = { score: newScore, prev: i, tokenLen: len };
          }
        }
      }

      // Byte fallback for single character
      if (best[i + 1] === null) {
        const bytes = new TextEncoder().encode(prefixed[i]);
        const byteScore = best[i].score - 10 * bytes.length;
        best[i + 1] = { score: byteScore, prev: i, tokenLen: 1, isBytes: true, bytes };
      }
    }

    // Backtrack to get tokens
    
    const tokens = [];
    let pos = n;
    while (pos > 0) {
      const state = best[pos];
      if (state.isBytes && state.bytes) {
        for (let j = state.bytes.length - 1; j >= 0; j--) {
          const byteId = this.#byteTokens.get(state.bytes[j]);
          tokens.push(byteId ?? this.#getUnkTokenId());
        }
      } else {
        const substr = prefixed.slice(state.prev, pos);
        const tokenId = this.#vocab.get(substr);
        tokens.push(tokenId ?? this.#getUnkTokenId());
      }
      pos = state.prev;
    }

    return tokens.reverse();
  }

  
  #encodeBPE(text) {
    if (text.length === 0) return [];

    let normalized = text;
    const ids = [];
    if (this.#addSpacePrefix && !normalized.startsWith(' ')) {
      normalized = ` ${normalized}`;
    }
    const pretokens = this.#splitBpePretokens(normalized);
    for (const pretoken of pretokens) {
      const encoded = this.#useByteLevelEncoding
        ? encodeByteLevelText(pretoken, this.#byteEncoder)
        : pretoken.replace(/ /g, this.#spacePrefixChar);
      if (this.#mergeRanks.size === 0) {
        ids.push(...this.#encodeBPEGreedy(encoded));
        continue;
      }
      for (const token of this.#bpeTokenize(encoded)) {
        const id = this.#vocab.get(token);
        if (id !== undefined) {
          ids.push(id);
          continue;
        }
        const bytes = new TextEncoder().encode(token);
        for (const b of bytes) {
          const byteId = this.#byteTokens.get(b);
          if (byteId !== undefined) {
            ids.push(byteId);
            continue;
          }
          const byteToken = `<0x${b.toString(16).padStart(2, '0').toUpperCase()}>`;
          ids.push(this.#vocab.get(byteToken) ?? this.#getUnkTokenId());
        }
      }
    }
    return ids;
  }

  #encodeWordPiece(text) {
    if (text.length === 0) return [];
    const words = this.#splitEveryCharacter
      ? Array.from(text)
      : text.match(/[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]/gu) ?? [];
    const ids = [];
    for (const word of words) {
      if (/^\s+$/u.test(word)) continue;
      if (word.length > this.#wordPieceMaxChars) {
        ids.push(this.#getUnkTokenId());
        continue;
      }
      let start = 0;
      const wordIds = [];
      let failed = false;
      while (start < word.length) {
        let end = word.length;
        let tokenId;
        while (start < end) {
          const piece = `${start > 0 ? this.#wordPiecePrefix : ''}${word.slice(start, end)}`;
          tokenId = this.#vocab.get(piece);
          if (tokenId !== undefined) break;
          end -= 1;
        }
        if (tokenId === undefined) {
          failed = true;
          break;
        }
        wordIds.push(tokenId);
        start = end;
      }
      ids.push(...(failed ? [this.#getUnkTokenId()] : wordIds));
    }
    return ids;
  }

  #splitBpePretokens(text) {
    if (!this.#useByteLevelEncoding || !this.#splitPretokenizer) return [text];
    const regex = new RegExp(this.#splitPretokenizer.source, this.#splitPretokenizer.flags);
    const pretokens = [];
    let cursor = 0;
    for (const match of text.matchAll(regex)) {
      const index = match.index ?? cursor;
      if (index > cursor) pretokens.push(text.slice(cursor, index));
      if (match[0]) pretokens.push(match[0]);
      cursor = index + match[0].length;
    }
    if (cursor < text.length) pretokens.push(text.slice(cursor));
    return pretokens.length > 0 ? pretokens : [text];
  }

  
  #encodeBPEGreedy(text) {
    
    const ids = [];
    let pos = 0;

    while (pos < text.length) {
      let bestLen = 0;
      let bestId = this.#getUnkTokenId();

      const maxLen = Math.min(32, text.length - pos);
      for (let len = maxLen; len >= 1; len--) {
        const substr = text.slice(pos, pos + len);
        const id = this.#vocab.get(substr);
        if (id !== undefined) {
          bestLen = len;
          bestId = id;
          break;
        }
      }

      if (bestLen === 0) {
        const char = text[pos];
        const bytes = new TextEncoder().encode(char);
        for (const b of bytes) {
          const byteId = this.#byteTokens.get(b);
          if (byteId !== undefined) {
            ids.push(byteId);
            continue;
          }
          const byteToken = `<0x${b.toString(16).padStart(2, '0').toUpperCase()}>`;
          ids.push(this.#vocab.get(byteToken) ?? this.#getUnkTokenId());
        }
        pos += 1;
      } else {
        ids.push(bestId);
        pos += bestLen;
      }
    }

    return ids;
  }

  
  #bpeTokenize(text) {
    if (text.length === 0) return [];

    let tokens = text.split('');
    if (tokens.length === 1) return tokens;

    while (tokens.length > 1) {
      let minRank = Infinity;
      
      let minPair = null;

      for (let i = 0; i < tokens.length - 1; i++) {
        const pair = `${tokens[i]} ${tokens[i + 1]}`;
        const rank = this.#mergeRanks.get(pair);
        if (rank !== undefined && rank < minRank) {
          minRank = rank;
          minPair = pair;
        }
      }

      if (!minPair) break;

      const [first, second] = minPair.split(' ');
      
      const newTokens = [];
      let i = 0;
      while (i < tokens.length) {
        if (i < tokens.length - 1 && tokens[i] === first && tokens[i + 1] === second) {
          newTokens.push(first + second);
          i += 2;
        } else {
          newTokens.push(tokens[i]);
          i += 1;
        }
      }
      tokens = newTokens;
    }

    return tokens;
  }

  
  decode(ids, skipSpecialTokens = true, trim = true) {
    if (this.#vocab.size === 0) {
      throw new Error('BundledTokenizer not loaded');
    }

    
    const tokens = [];
    for (const id of ids) {
      if (skipSpecialTokens && this.isSpecialToken(id)) {
        continue;
      }

      const token = this.#reverseVocab.get(id);
      if (token !== undefined) {
        tokens.push(token);
      }
    }

    let result;
    if (this.#type === 'bpe' && this.#useByteLevelEncoding) {
      result = decodeByteLevelTokens(tokens, this.#byteDecoder);
    } else if (this.#type === 'wordpiece') {
      if (this.#splitEveryCharacter) {
        result = tokens.join('');
      } else {
        result = tokens.reduce((text, token) => {
          if (token.startsWith(this.#wordPiecePrefix)) {
            return `${text}${token.slice(this.#wordPiecePrefix.length)}`;
          }
          return text ? `${text} ${token}` : token;
        }, '');
      }
    } else {
      // Join and convert ▁ back to spaces, handle GPT-style markers
      result = decodeByteFallbackTokens(tokens)
        .replace(/▁/g, ' ')
        .replace(/Ġ/g, ' ')
        .replace(/Ċ/g, '\n');
    }

    // Only trim when requested (not during streaming where spaces matter)
    return trim ? result.trim() : result;
  }

  getHotTokenIds(limit) {
    const resolvedLimit = Math.trunc(Number(limit));
    if (!Number.isFinite(resolvedLimit) || resolvedLimit <= 0) {
      return [];
    }
    if (this.#type === 'bpe' && (!Array.isArray(this.#scores) || this.#scores.length === 0)) {
      return rankFallbackBpeHotTokenIds(
        this.#reverseVocab,
        resolvedLimit,
        (tokenId) => this.isSpecialToken(tokenId)
      );
    }
    if (!Array.isArray(this.#scores) || this.#scores.length === 0) {
      return null;
    }
    const ranked = [];
    for (let id = 0; id < this.#scores.length; id += 1) {
      if (this.isSpecialToken(id)) {
        continue;
      }
      ranked.push({ id, score: Number(this.#scores[id] ?? 0) });
    }
    ranked.sort((a, b) => b.score - a.score || a.id - b.id);
    return ranked.slice(0, resolvedLimit).map((entry) => entry.id);
  }
}
