import patterns from './tokenizer-pre.json' with { type: 'json' };

/** @type {Readonly<Record<string, string>>} */
const declaredPatterns = patterns;

/** @type {typeof import('./tokenizer.js').buildGGUFTokenizer} */
export function buildGGUFTokenizer(metadata) {
  const tokens = metadata['tokenizer.ggml.tokens'];
  if (tokens === undefined) return null;
  const pre = metadata['tokenizer.ggml.pre'];
  const regex = typeof pre === 'string' ? declaredPatterns[pre] : undefined;
  if (metadata['tokenizer.ggml.model'] !== 'gpt2' || !regex) {
    throw new Error(`Unsupported native GGUF tokenizer model/pre: ${metadata['tokenizer.ggml.model']}/${pre}.`);
  }
  const types = metadata['tokenizer.ggml.token_type'];
  const merges = metadata['tokenizer.ggml.merges'];
  if (!Array.isArray(tokens) || !tokens.length || tokens.some(token => typeof token !== 'string')
      || new Set(tokens).size !== tokens.length || !Array.isArray(types) || types.length !== tokens.length
      || !Array.isArray(merges) || merges.some(merge => typeof merge !== 'string')) {
    throw new Error('Malformed native GGUF BPE vocabulary, token types or merges.');
  }
  const vocab = Object.fromEntries(tokens.map((token, id) => [token, id]));
  const addedTokens = tokens.flatMap((content, id) => types[id] === 3 || types[id] === 4
    ? [{ id, content, single_word: false, lstrip: false, rstrip: false, normalized: false, special: types[id] === 3 }] : []);
  const tokenizerConfig = {
    vocab_size: tokens.length,
    bos_token_id: metadata['tokenizer.ggml.bos_token_id'],
    eos_token_id: metadata['tokenizer.ggml.eos_token_id'],
    pad_token_id: metadata['tokenizer.ggml.padding_token_id'],
    add_bos_token: metadata['tokenizer.ggml.add_bos_token'] === true,
    add_eos_token: metadata['tokenizer.ggml.add_eos_token'] === true,
    chat_template: metadata['tokenizer.chat_template'],
  };
  for (const key of /** @type {const} */ (['bos_token_id', 'eos_token_id', 'pad_token_id'])) {
    const value = tokenizerConfig[key];
    if (value !== undefined && (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0 || value >= tokens.length)) {
      throw new Error(`GGUF ${key} is outside the source vocabulary.`);
    }
  }
  return { tokenizerConfig, tokenizerJson: {
    version: '1.0', truncation: null, padding: null, normalizer: null,
    added_tokens: addedTokens,
    pre_tokenizer: { type: 'Sequence', pretokenizers: [
      { type: 'Split', pattern: { Regex: regex }, behavior: 'Isolated', invert: false },
      { type: 'ByteLevel', add_prefix_space: false, trim_offsets: true, use_regex: false },
    ] },
    post_processor: null, decoder: { type: 'ByteLevel', add_prefix_space: false, trim_offsets: true, use_regex: false },
    model: { type: 'BPE', dropout: null, unk_token: null, continuing_subword_prefix: '', end_of_word_suffix: '', fuse_unk: false, byte_fallback: false, vocab, merges },
  } };
}
