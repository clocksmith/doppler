// Shared helpers for the reproducible builders. None of the generated programs
// imports this file.
import fs from 'node:fs';

// Text with CRLF and CR line ends normalized to LF.
export function readText(file) {
  return fs.readFileSync(file, 'utf8').replace(/\r\n?/g, '\n');
}

export function writeText(file, text) {
  fs.writeFileSync(file, text, 'utf8');
}

export function asciiBytes(text) {
  const result = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code > 127) throw new Error(`Non-ASCII character ${code} at offset ${i}.`);
    result[i] = code;
  }
  return result;
}

const SHORT_ESCAPES = { '"': '\\"', '\\': '\\\\', '\n': '\\n', '\r': '\\r', '\t': '\\t', '\b': '\\b', '\f': '\\f' };

function quote(text) {
  let result = '"';
  for (const ch of text) {
    const code = ch.codePointAt(0);
    if (Object.hasOwn(SHORT_ESCAPES, ch)) result += SHORT_ESCAPES[ch];
    else if (code >= 0x20 && code <= 0x7e) result += ch;
    else {
      for (let i = 0; i < ch.length; i++) result += '\\u' + ch.charCodeAt(i).toString(16).padStart(4, '0');
    }
  }
  return result + '"';
}

// JSON with every non-printable or non-ASCII character escaped. Without an
// indent, items are separated by `itemSeparator` (", " or ","); with an indent,
// each item sits on its own line, matching the committed generated files.
export function jsonText(value, { indent = null, itemSeparator = ', ', keySeparator = ': ' } = {}) {
  const separator = indent === null ? itemSeparator : ',';
  const encode = (item, level) => {
    if (item === null) return 'null';
    if (item === true) return 'true';
    if (item === false) return 'false';
    if (typeof item === 'number') {
      if (!Number.isInteger(item)) throw new Error(`Only integers are emitted: ${item}`);
      return String(item);
    }
    if (typeof item === 'string') return quote(item);
    const entries = Array.isArray(item) ? item.map((element) => encode(element, level + 1))
      : Object.entries(item).map(([key, element]) => quote(key) + keySeparator + encode(element, level + 1));
    const [open, close] = Array.isArray(item) ? ['[', ']'] : ['{', '}'];
    if (!entries.length) return open + close;
    if (indent === null) return open + entries.join(separator) + close;
    const inner = '\n' + ' '.repeat(indent * (level + 1));
    return open + inner + entries.join(separator + inner) + '\n' + ' '.repeat(indent * level) + close;
  };
  return encode(value, 0);
}
