function assertPlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function requireToken(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value;
}

function validateRelativeFile(value) {
  const file = requireToken(value, 'sourceTokenizer.vocabFile');
  if (file.startsWith('/') || file.includes('\\') || file.split('/').includes('..')) {
    throw new Error('sourceTokenizer.vocabFile must stay inside the source model directory.');
  }
  return file;
}

function parseVocab(text) {
  if (typeof text !== 'string') {
    throw new Error('Source vocabulary must be text.');
  }
  const tokens = text.replace(/\r/gu, '').split('\n');
  if (tokens.at(-1) === '') tokens.pop();
  if (tokens.length === 0 || tokens.some((token) => token.length === 0)) {
    throw new Error('Source vocabulary must contain one non-empty token per line.');
  }
  const vocab = {};
  for (let id = 0; id < tokens.length; id += 1) {
    const token = tokens[id];
    if (Object.prototype.hasOwnProperty.call(vocab, token)) {
      throw new Error(`Source vocabulary contains duplicate token "${token}".`);
    }
    vocab[token] = id;
  }
  return { tokens, vocab };
}

export function validateSourceTokenizerPolicy(policy) {
  if (policy == null) return null;
  const value = assertPlainObject(policy, 'sourceTokenizer');
  if (value.kind !== 'character_vocab' && value.kind !== 'greedy_vocab') {
    throw new Error(`Unsupported sourceTokenizer.kind "${String(value.kind)}".`);
  }
  const specialTokens = assertPlainObject(value.specialTokens, 'sourceTokenizer.specialTokens');
  const normalized = {
    kind: value.kind,
    vocabFile: validateRelativeFile(value.vocabFile),
    specialTokens: {
      pad: requireToken(specialTokens.pad, 'sourceTokenizer.specialTokens.pad'),
      bos: requireToken(specialTokens.bos, 'sourceTokenizer.specialTokens.bos'),
      eos: value.kind === 'character_vocab'
        ? requireToken(specialTokens.eos, 'sourceTokenizer.specialTokens.eos')
        : specialTokens.eos == null
          ? null
          : requireToken(specialTokens.eos, 'sourceTokenizer.specialTokens.eos'),
      unk: requireToken(specialTokens.unk, 'sourceTokenizer.specialTokens.unk'),
      mask: specialTokens.mask == null
        ? null
        : requireToken(specialTokens.mask, 'sourceTokenizer.specialTokens.mask'),
    },
    addBosToken: value.addBosToken === true,
    addEosToken: value.addEosToken === true,
  };
  if (typeof value.addBosToken !== 'boolean' || typeof value.addEosToken !== 'boolean') {
    throw new Error('sourceTokenizer.addBosToken and addEosToken must be booleans.');
  }
  return normalized;
}

export function buildSourceTokenizerJson(policy, vocabText) {
  const normalized = validateSourceTokenizerPolicy(policy);
  if (!normalized) throw new Error('sourceTokenizer policy is required.');
  const { tokens, vocab } = parseVocab(vocabText);
  for (const [role, token] of Object.entries(normalized.specialTokens)) {
    if (token != null && !Object.prototype.hasOwnProperty.call(vocab, token)) {
      throw new Error(`Source vocabulary is missing ${role} token "${token}".`);
    }
  }
  const uniqueSpecialTokens = [...new Set(Object.values(normalized.specialTokens).filter(Boolean))];
  const characterVocab = normalized.kind === 'character_vocab';
  return {
    version: '1.0',
    truncation: null,
    padding: null,
    added_tokens: uniqueSpecialTokens.map((token) => ({
      id: vocab[token],
      content: token,
      single_word: false,
      lstrip: false,
      rstrip: false,
      normalized: false,
      special: true,
    })),
    normalizer: null,
    pre_tokenizer: characterVocab
      ? {
        type: 'Split',
        pattern: { String: '' },
        behavior: 'Removed',
        invert: false,
      }
      : null,
    post_processor: null,
    decoder: characterVocab
      ? {
        type: 'WordPiece',
        prefix: '##',
        cleanup: false,
      }
      : null,
    model: characterVocab
      ? {
        type: 'WordPiece',
        dropout: null,
        unk_token: normalized.specialTokens.unk,
        continuing_subword_prefix: '##',
        max_input_chars_per_word: 100,
        vocab,
      }
      : {
        type: 'BPE',
        dropout: null,
        unk_token: normalized.specialTokens.unk,
        continuing_subword_prefix: null,
        end_of_word_suffix: null,
        fuse_unk: false,
        byte_fallback: false,
        vocab,
        merges: [],
      },
    special_tokens_map: {
      pad_token: normalized.specialTokens.pad,
      bos_token: normalized.specialTokens.bos,
      ...(normalized.specialTokens.eos ? { eos_token: normalized.specialTokens.eos } : {}),
      unk_token: normalized.specialTokens.unk,
      ...(normalized.specialTokens.mask ? { mask_token: normalized.specialTokens.mask } : {}),
    },
    add_bos_token: normalized.addBosToken,
    add_eos_token: normalized.addEosToken,
    source_vocab_size: tokens.length,
  };
}

export function buildCharacterTokenizerJson(policy, vocabText) {
  const normalized = validateSourceTokenizerPolicy(policy);
  if (normalized?.kind !== 'character_vocab') {
    throw new Error('buildCharacterTokenizerJson requires sourceTokenizer.kind="character_vocab".');
  }
  return buildSourceTokenizerJson(normalized, vocabText);
}
