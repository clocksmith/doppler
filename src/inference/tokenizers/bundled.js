

import { BaseTokenizer } from './base.js';
import { log } from '../../debug/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';

function pickCandidate(...values) {
  for (const value of values) {
    if (value != null) return value;
  }
  return null;
}

function resolveTokenId(value, vocab, label) {
  if (value == null) return null;
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const id = vocab.get(value);
    if (id === undefined) {
      throw new Error(`[Tokenizer] Special token "${label}" not found in vocab: "${value}"`);
    }
    return id;
  }
  return null;
}

function resolveSpecialTokens(specialTokensRaw, fallbackTokens, vocab) {
  const padCandidate = pickCandidate(
    specialTokensRaw?.pad,
    specialTokensRaw?.pad_token,
    specialTokensRaw?.pad_token_id,
    fallbackTokens?.pad
  );
  const bosCandidate = pickCandidate(
    specialTokensRaw?.bos,
    specialTokensRaw?.bos_token,
    specialTokensRaw?.bos_token_id,
    fallbackTokens?.bos
  );
  const eosCandidate = pickCandidate(
    specialTokensRaw?.eos,
    specialTokensRaw?.eos_token,
    specialTokensRaw?.eos_token_id,
    fallbackTokens?.eos
  );
  const unkCandidate = pickCandidate(
    specialTokensRaw?.unk,
    specialTokensRaw?.unk_token,
    specialTokensRaw?.unk_token_id,
    fallbackTokens?.unk
  );

  const resolved = {
    pad: resolveTokenId(padCandidate, vocab, 'pad'),
    bos: resolveTokenId(bosCandidate, vocab, 'bos'),
    eos: resolveTokenId(eosCandidate, vocab, 'eos'),
    unk: resolveTokenId(unkCandidate, vocab, 'unk'),
  };

  if (resolved.eos == null) {
    throw new Error('[Tokenizer] Missing EOS token in tokenizer.json and runtime config.');
  }

  return resolved;
}


export class TransformersTokenizer extends BaseTokenizer {
  
  #tokenizer = null;
  
  #modelId;

  
  constructor(config = {}) {
    super(config);
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
}


export class BundledTokenizer extends BaseTokenizer {
  
  #vocab = new Map();
  
  #reverseVocab = new Map();
  
  #merges = [];
  
  #mergeRanks = new Map();
  
  #scores = [];
  
  #tokenTypes = [];
  
  #type = 'bpe';
  
  #byteTokens = new Map();
  
  #specialTokenPatterns = [];
  
  #specialTokenIds = new Set();
  
  #addSpacePrefix = true;
  
  #spacePrefixChar = '▁';

  
  constructor(config = {}) {
    super(config);
  }

  
  isSpecialToken(tokenId) {
    if (this.#specialTokenIds.size > 0) {
      return this.#specialTokenIds.has(tokenId);
    }
    return super.isSpecialToken(tokenId);
  }

  
  load(tokenizerJson) {
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
    this.#type =  (model.type?.toLowerCase()) || 'bpe';
    log.info('Tokenizer', `HuggingFace model.type="${model.type}", using type="${this.#type}"`);
    let maxId = -1;

    // Handle vocab based on type
    if (this.#type === 'unigram' && Array.isArray(model.vocab)) {
      // Unigram format: [[token, score], ...]
      for (let i = 0; i < model.vocab.length; i++) {
        const [token, score] = model.vocab[i];
        this.#vocab.set(token, i);
        this.#reverseVocab.set(i, token);
        this.#scores.push(score);
        if (i > maxId) maxId = i;

        // Track byte tokens
        if (token.match(/^<0x[0-9A-Fa-f]{2}>$/)) {
          const byteVal = parseInt(token.slice(3, 5), 16);
          this.#byteTokens.set(byteVal, i);
        }
      }
    } else {
      // BPE format: { token: id }
      for (const [token, id] of Object.entries(model.vocab || {})) {
        const numId = typeof id === 'number' ? id : parseInt( (id), 10);
        this.#vocab.set(token, numId);
        this.#reverseVocab.set(numId, token);
        if (Number.isFinite(numId) && numId > maxId) maxId = numId;

        // Track byte tokens
        if (token.match(/^<0x[0-9A-Fa-f]{2}>$/)) {
          const byteVal = parseInt(token.slice(3, 5), 16);
          this.#byteTokens.set(byteVal, numId);
        }
      }
    }

    this.vocabSize = this.#vocab.size;

    // Load merges from model.merges
    if (model.merges && model.merges.length > 0) {
      this.#merges = model.merges;
      for (let i = 0; i < this.#merges.length; i++) {
        this.#mergeRanks.set(this.#merges[i], i);
      }
    }

    // Extract special tokens from added_tokens
    this.specialTokens = {
      pad: model.pad_id ?? 0,
      bos: 1,
      eos: 2,
      unk: model.unk_id ?? 0,
    };
    this.#specialTokenIds = new Set();

    for (const token of hf.added_tokens || []) {
      if (token.special) {
        const content = token.content;
        const id = typeof token.id === 'number' ? token.id : parseInt( (token.id), 10);
        if (Number.isFinite(id) && id > maxId) maxId = id;
        if (Number.isFinite(id)) {
          this.#specialTokenIds.add(id);
        }
        // Add to vocab if not already there
        if (!this.#vocab.has(content)) {
          this.#vocab.set(content, id);
          this.#reverseVocab.set(id, content);
        }
        // Store for pattern matching during encode (skip single-char tokens)
        if (content.length > 1) {
          this.#specialTokenPatterns.push({ content, id });
        }
        // Identify special token types
        if (content === '<bos>' || content === '<s>' || content.includes('bos')) {
          this.specialTokens.bos = id;
        } else if (content === '<eos>' || content === '</s>' || content.includes('eos')) {
          this.specialTokens.eos = id;
        } else if (content === '<pad>' || content.includes('pad')) {
          this.specialTokens.pad = id;
        } else if (content === '<unk>' || content.includes('unk')) {
          this.specialTokens.unk = id;
        }
      }
    }
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
    this.addBosToken = hf.add_bos_token ?? runtimeDefaults.addBosToken;
    this.addEosToken = hf.add_eos_token ?? runtimeDefaults.addEosToken;
    // NOTE: Default to FALSE for add_prefix_space - HuggingFace tokenizers typically
    // don't add a space prefix to the first token. Models like Gemma, Llama use
    // the normalizer/pre_tokenizer to handle spaces within text, not at the start.
    this.#addSpacePrefix = model.add_prefix_space ?? model.add_dummy_prefix ?? false;

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
    this.#type =  (tokenizerJson.type) || 'bpe';

    // Build vocab maps
    for (const [token, id] of Object.entries(tokenizerJson.vocab)) {
      const numId = typeof id === 'number' ? id : parseInt( (id), 10);
      this.#vocab.set(token, numId);
      this.#reverseVocab.set(numId, token);

      // Track byte tokens for fallback
      if (token.match(/^<0x[0-9A-Fa-f]{2}>$/)) {
        const byteVal = parseInt(token.slice(3, 5), 16);
        this.#byteTokens.set(byteVal, numId);
      }
    }

    this.vocabSize = this.#vocab.size;

    // Load merges for BPE
    if (tokenizerJson.merges && tokenizerJson.merges.length > 0) {
      this.#merges = tokenizerJson.merges;
      for (let i = 0; i < this.#merges.length; i++) {
        this.#mergeRanks.set(this.#merges[i], i);
      }
    }

    // Load scores for Unigram
    if (tokenizerJson.scores && tokenizerJson.scores.length > 0) {
      this.#scores = tokenizerJson.scores;
    }

    // Load token types if available
    if (tokenizerJson.tokenTypes) {
      this.#tokenTypes = tokenizerJson.tokenTypes;
    }

    // Set special tokens - support both camelCase and snake_case formats
    const specialTokensRaw =  (tokenizerJson.specialTokens ||  (tokenizerJson).special_tokens);
    this.specialTokens = resolveSpecialTokens(specialTokensRaw, this.specialTokens, this.#vocab);
    log.debug('Tokenizer', `Special tokens: BOS=${this.specialTokens.bos}, EOS=${this.specialTokens.eos}`);
    this.#specialTokenIds = new Set();
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

    const runtimeDefaults = getRuntimeConfig().inference.tokenizer;
    this.addBosToken = tokenizerJson.addBosToken ?? runtimeDefaults.addBosToken;
    this.addEosToken = tokenizerJson.addEosToken ?? runtimeDefaults.addEosToken;
    if (this.addBosToken && this.specialTokens.bos == null) {
      throw new Error('[Tokenizer] addBosToken is enabled but bos token is missing.');
    }
    if (this.addEosToken && this.specialTokens.eos == null) {
      throw new Error('[Tokenizer] addEosToken is enabled but eos token is missing.');
    }
    // NOTE: Default to FALSE - first word shouldn't get space prefix
    // Space prefixes are only for words that follow a space in original text
    this.#addSpacePrefix = tokenizerJson.addSpacePrefix === true;

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
      ids.push(this.specialTokens.bos ?? 1);
    }

    // Split text around special tokens and tokenize each segment
    const segments = this.#splitOnSpecialTokens(text);
    for (const seg of segments) {
      if (seg.isSpecial && seg.id !== undefined) {
        ids.push(seg.id);
      } else if (seg.text && seg.text.length > 0) {
        if (this.#type === 'unigram') {
          ids.push(...this.#encodeUnigram(seg.text));
        } else {
          ids.push(...this.#encodeBPE(seg.text));
        }
      }
    }

    if (this.addEosToken) {
      ids.push(this.specialTokens.eos ?? 2);
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
    // Normalize: convert spaces to the model's space prefix character
    // This turns "The color of" into "The▁color▁of"
    // Note: we do NOT add an extra prefix at the start - the first word has no prefix
    const sp = this.#spacePrefixChar;
    const prefixed = text.replace(/ /g, sp);

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
          tokens.push(byteId ?? (this.specialTokens.unk ?? 0));
        }
      } else {
        const substr = prefixed.slice(state.prev, pos);
        const tokenId = this.#vocab.get(substr);
        tokens.push(tokenId ?? (this.specialTokens.unk ?? 0));
      }
      pos = state.prev;
    }

    return tokens.reverse();
  }

  
  #encodeBPE(text) {
    if (text.length === 0) return [];

    let normalized = text;
    if (this.#addSpacePrefix && !normalized.startsWith(' ')) {
      normalized = ` ${normalized}`;
    }
    const sp = this.#spacePrefixChar;
    const prefixed = normalized.replace(/ /g, sp);

    if (this.#mergeRanks.size === 0) {
      return this.#encodeBPEGreedy(prefixed);
    }

    const tokens = this.#bpeTokenize(prefixed);
    
    const ids = [];
    for (const token of tokens) {
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
        ids.push(this.#vocab.get(byteToken) ?? (this.specialTokens.unk ?? 0));
      }
    }

    return ids;
  }

  
  #encodeBPEGreedy(text) {
    
    const ids = [];
    let pos = 0;

    while (pos < text.length) {
      let bestLen = 0;
      let bestId = this.specialTokens.unk ?? 0;

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
          ids.push(this.#vocab.get(byteToken) ?? (this.specialTokens.unk ?? 0));
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
        // Handle byte tokens
        if (token.match(/^<0x[0-9A-Fa-f]{2}>$/)) {
          const byteVal = parseInt(token.slice(3, 5), 16);
          tokens.push(String.fromCharCode(byteVal));
        } else {
          tokens.push(token);
        }
      }
    }

    // Join and convert ▁ back to spaces, handle GPT-style markers
    let result = tokens.join('')
      .replace(/▁/g, ' ')
      .replace(/Ġ/g, ' ')
      .replace(/Ċ/g, '\n');

    // Only trim when requested (not during streaming where spaces matter)
    return trim ? result.trim() : result;
  }
}
