import { parseByteTokenValue } from './decoders.js';

// Keep a potentially valid, incomplete UTF-8 suffix, at most three bytes.
function incompleteStart(bytes) {
  for (let size = 1; size <= Math.min(3, bytes.length); size++) {
    const start = bytes.length - size;
    const lead = bytes[start];
    const width = lead >= 0xc2 && lead <= 0xdf ? 2
      : lead >= 0xe0 && lead <= 0xef ? 3 : lead >= 0xf0 && lead <= 0xf4 ? 4 : 0;
    if (!width || size >= width) continue;
    let valid = true;
    for (let i = start + 1; i < bytes.length; i++) {
      const byte = bytes[i];
      if (byte < 0x80 || byte > 0xbf
        || (i === start + 1 && ((lead === 0xe0 && byte < 0xa0) || (lead === 0xed && byte > 0x9f)
          || (lead === 0xf0 && byte < 0x90) || (lead === 0xf4 && byte > 0x8f)))) valid = false;
    }
    if (valid) return start;
  }
  return bytes.length;
}

function createUtf8Decoder(snapshot = null) {
  let decoder = new TextDecoder('utf-8', { fatal: false });
  let pending = snapshot ? Uint8Array.from(snapshot.pending) : new Uint8Array();
  let started = snapshot?.started === true;
  // Recreate only the finite decoder control state, never the full token history.
  // A harmless prefix consumes the BOM flag before reinstating incomplete bytes.
  if (started) decoder.decode(Uint8Array.of(0x20), { stream: true });
  if (pending.length) decoder.decode(pending, { stream: true });
  return {
    push(bytes) {
      const combined = new Uint8Array(pending.length + bytes.length);
      combined.set(pending);
      combined.set(bytes, pending.length);
      const end = incompleteStart(combined);
      pending = combined.slice(end);
      const text = decoder.decode(Uint8Array.from(bytes), { stream: true });
      started ||= end > 0;
      return text;
    },
    snapshot() { return { pending: [...pending], started }; },
    pendingText() { return pending.length ? '\ufffd' : ''; },
    finish() {
      const text = decoder.decode();
      pending = new Uint8Array();
      started = false;
      decoder = new TextDecoder('utf-8', { fatal: false });
      return text;
    },
  };
}

// Every call owns its state; cached vocabularies never own stream state.
export function createBundledIncrementalDecoder(options) {
  return createDecoder(options, null);
}

function createDecoder(options, snapshot) {
  const { tokenForId, byteLevel, byteDecoder, wordPiece, splitEveryCharacter, wordPiecePrefix } = options;
  const utf8 = createUtf8Decoder(snapshot?.utf8);
  let hasText = snapshot?.hasText === true;
  let closed = snapshot?.closed === true;
  const markers = value => value.replace(/▁/g, ' ').replace(/Ġ/g, ' ').replace(/Ċ/g, '\n');
  return {
    snapshot() { return { schema: 'doppler.bundled-decoder-state/v1', utf8: utf8.snapshot(), hasText, closed }; },
    clone() { return createDecoder(options, this.snapshot()); },
    push(tokenId) {
      if (closed) throw new Error('Incremental tokenizer decoder is closed.');
      const token = tokenForId(tokenId);
      if (token === undefined) return '';
      if (wordPiece) {
        const piece = splitEveryCharacter ? token : token.startsWith(wordPiecePrefix)
          ? token.slice(wordPiecePrefix.length) : `${hasText ? ' ' : ''}${token}`;
        hasText ||= piece.length > 0;
        return piece;
      }
      const byte = parseByteTokenValue(token);
      if (byteLevel) {
        const bytes = [];
        if (byte !== null) bytes.push(byte);
        else for (const char of token) {
          const mapped = byteDecoder.get(char);
          if (mapped !== undefined) bytes.push(mapped);
          else for (const value of new TextEncoder().encode(char)) bytes.push(value);
        }
        return utf8.push(bytes).replace(/▁/g, ' ');
      }
      return markers(byte === null ? utf8.finish() + token : utf8.push([byte]));
    },
    // Prefix decoding flushes unfinished bytes. Only stop matching uses this
    // preview; display text waits for the complete character or final flush.
    pendingText() { return utf8.pendingText(); },
    finish() {
      if (closed) throw new Error('Incremental tokenizer decoder is closed.');
      closed = true;
      return byteLevel ? utf8.finish().replace(/▁/g, ' ') : markers(utf8.finish());
    },
  };
}
