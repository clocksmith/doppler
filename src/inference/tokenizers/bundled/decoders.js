function hexNibble(code) {
  if (code >= 48 && code <= 57) return code - 48;
  if (code >= 65 && code <= 70) return code - 55;
  if (code >= 97 && code <= 102) return code - 87;
  return -1;
}

export function resolveByteLevelPretokenizerConfig(preTokenizer) {
  if (!preTokenizer || typeof preTokenizer !== 'object') {
    return {
      useByteLevel: false,
      addPrefixSpace: null,
    };
  }

  if (preTokenizer.type === 'ByteLevel') {
    return {
      useByteLevel: true,
      addPrefixSpace: preTokenizer.add_prefix_space === true,
    };
  }

  if (preTokenizer.type === 'Sequence' && Array.isArray(preTokenizer.pretokenizers)) {
    for (const entry of preTokenizer.pretokenizers) {
      const resolved = resolveByteLevelPretokenizerConfig(entry);
      if (resolved.useByteLevel) {
        return resolved;
      }
    }
  }

  return {
    useByteLevel: false,
    addPrefixSpace: null,
  };
}

export function parseByteTokenValue(token) {
  if (
    typeof token !== 'string'
    || token.length !== 6
    || token.charCodeAt(0) !== 60
    || token.charCodeAt(1) !== 48
    || token.charCodeAt(2) !== 120
    || token.charCodeAt(5) !== 62
  ) {
    return null;
  }
  const hi = hexNibble(token.charCodeAt(3));
  const lo = hexNibble(token.charCodeAt(4));
  return hi >= 0 && lo >= 0 ? (hi << 4) | lo : null;
}

export function decodeByteFallbackTokens(tokens) {
  const parts = [];
  let bytes = [];
  const flushBytes = () => {
    if (bytes.length === 0) return;
    parts.push(new TextDecoder('utf-8', { fatal: false }).decode(Uint8Array.from(bytes)));
    bytes = [];
  };

  for (const token of tokens) {
    const byteValue = parseByteTokenValue(token);
    if (byteValue !== null) {
      bytes.push(byteValue);
      continue;
    }
    flushBytes();
    parts.push(token);
  }
  flushBytes();
  return parts.join('');
}

export function createByteLevelCodec() {
  const base = [];
  for (let i = 33; i <= 126; i++) base.push(i);
  for (let i = 161; i <= 172; i++) base.push(i);
  for (let i = 174; i <= 255; i++) base.push(i);

  const chars = [...base];
  let extra = 0;
  for (let byte = 0; byte <= 255; byte++) {
    if (!base.includes(byte)) {
      base.push(byte);
      chars.push(256 + extra);
      extra += 1;
    }
  }

  const decoder = new Map();
  const encoder = new Map();
  for (let i = 0; i < base.length; i++) {
    decoder.set(String.fromCodePoint(chars[i]), base[i]);
    encoder.set(base[i], String.fromCodePoint(chars[i]));
  }
  return { decoder, encoder };
}

export function encodeByteLevelText(text, encoder) {
  const bytes = new TextEncoder().encode(text);
  let result = '';
  for (const byte of bytes) {
    result += encoder?.get(byte) ?? String.fromCharCode(byte);
  }
  return result;
}

export function decodeByteLevelTokens(tokens, decoder) {
  const bytes = [];
  for (const token of tokens) {
    const byteValue = parseByteTokenValue(token);
    if (byteValue !== null) {
      bytes.push(byteValue);
      continue;
    }
    for (const char of token) {
      const mapped = decoder.get(char);
      if (mapped != null) {
        bytes.push(mapped);
        continue;
      }
      const fallbackBytes = new TextEncoder().encode(char);
      for (const byte of fallbackBytes) {
        bytes.push(byte);
      }
    }
  }
  return new TextDecoder('utf-8', { fatal: false })
    .decode(Uint8Array.from(bytes))
    .replace(/▁/g, ' ');
}
