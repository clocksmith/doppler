
function parseLine(line, lineNumber) {
  if (!line.trim()) {
    return null;
  }
  try {
    return JSON.parse(line);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`Invalid JSONL at line ${lineNumber}: ${message}`);
  }
}

export function parseJsonl(text) {
  const records = [];
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const value = parseLine(lines[i], i + 1);
    if (value !== null) {
      records.push(value);
    }
  }
  return records;
}

export async function loadJsonl(source, options = {}) {
  if (typeof source !== 'string') {
    throw new Error('loadJsonl expects a string source');
  }

  const looksLikeUrl = /^https?:\/\//i.test(source);
  const text = looksLikeUrl
    ? await (options.fetch ? options.fetch(source) : fetch(source).then((res) => res.text()))
    : source;
  return parseJsonl(text);
}

export function mapJsonl(records, mapper) {
  if (!mapper) {
    return records;
  }
  const mapped = [];
  for (const record of records) {
    const next = mapper(record);
    if (next !== null && next !== undefined) {
      mapped.push(next);
    }
  }
  return mapped;
}
