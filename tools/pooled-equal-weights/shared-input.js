// Pooled's native encode() handles ordinary BPE text; special tokens are explicit.
export function encodeSharedPrompt(tokenizer, prompt, format) {
  const text = format.before + prompt + format.after;
  const escaped = format.specialTokens.map(token => token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const pattern = new RegExp(`(${escaped.join('|')})`, 'g');
  const specials = new Set(format.specialTokens);
  return text.split(pattern).flatMap(part => {
    if (!specials.has(part)) return tokenizer.encode(part);
    const id = tokenizer.vocab[part];
    if (!Number.isSafeInteger(id) || id < 0) throw new Error(`Missing special token: ${part}`);
    return [id];
  });
}
