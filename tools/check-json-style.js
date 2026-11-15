export const RULE_IDS = Object.freeze(["JSON-001", "JSON-002", "JSON-003"]);

// Format checks only; domain schemas and contract validators remain authoritative.
// tsconfig is JSONC by contract. All other .json assets require strict JSON.
function jsonText(source, filePath) {
  if (!/(^|\/)tsconfig(?:\.[^/]+)?\.json$/.test(filePath)) return source;
  return source.replace(/"(?:\\[\s\S]|[^"\\])*"|\/\/[^\r\n]*|\/\*[\s\S]*?\*\//g, (token) => (
    token.startsWith('/') ? token.replace(/[^\r\n]/g, ' ') : token
  )).replace(/"(?:\\[\s\S]|[^"\\])*"|,(\s*[}\]])/g, (token, suffix) => (
    suffix === undefined ? token : ` ${suffix}`
  ));
}

// Sticky match at the token end: a following ':' makes the string a member name.
const MEMBER_COLON = /\s*:/y;
function isMemberName(text, end) {
  MEMBER_COLON.lastIndex = end;
  return MEMBER_COLON.test(text);
}

// JSON.parse turns an overflowing number into Infinity; finding one in the parsed
// value is how the number walk below knows it has work to do.
function hasNonFiniteNumber(value) {
  const pending = [value];
  while (pending.length > 0) {
    const item = pending.pop();
    if (typeof item === 'number') {
      if (!Number.isFinite(item)) return true;
    } else if (item !== null && typeof item === 'object') {
      for (const child of Array.isArray(item) ? item : Object.values(item)) pending.push(child);
    }
  }
  return false;
}

const MEMBER_TOKENS = /"(?:\\[\s\S]|[^"\\])*"|[{}\[\]]/g;
const ALL_TOKENS = /"(?:\\[\s\S]|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[{}\[\]]/g;

export function checkJsonSource(source, filePath = '') {
  const text = jsonText(source, filePath);
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    return [{ ruleId: 'JSON-001', message: `Invalid JSON: ${error.message}` }];
  }
  // Most tracked JSON is numeric reports; skip number tokens unless the parsed
  // value holds an overflow. A duplicate member can hide one, so rescan then.
  const fast = scanTokens(text, hasNonFiniteNumber(parsed) ? ALL_TOKENS : MEMBER_TOKENS);
  return fast.some(issue => issue.ruleId === 'JSON-002') ? scanTokens(text, ALL_TOKENS) : fast;
}

function scanTokens(text, tokens) {
  const issues = [];
  const stack = [];
  // JSON.parse establishes grammar first. This token walk preserves members that
  // JSON.parse would silently overwrite, including escaped equivalent key names.
  // Line numbers are only needed for issues, so count newlines lazily.
  let line = 1;
  let counted = 0;
  const lineAt = (index) => {
    for (; counted < index; counted += 1) {
      if (text.charCodeAt(counted) === 10) line += 1;
    }
    return line;
  };
  for (const match of text.matchAll(tokens)) {
    const token = match[0];
    if (token === '{' || token === '[') {
      stack.push(token === '{' ? new Set() : null);
    } else if (token === '}' || token === ']') {
      stack.pop();
    } else if (token.startsWith('"') && isMemberName(text, match.index + token.length)) {
      const keys = stack[stack.length - 1];
      const key = JSON.parse(token);
      if (keys.has(key)) {
        issues.push({ ruleId: 'JSON-002', line: lineAt(match.index), message: `Duplicate object member ${JSON.stringify(key)}; JSON.parse silently overwrites it.` });
      }
      keys.add(key);
    } else if (/^-?\d/.test(token) && !Number.isFinite(Number(token))) {
      issues.push({ ruleId: 'JSON-003', line: lineAt(match.index), message: `Number ${token} overflows JavaScript's finite numeric range.` });
    }
  }
  return issues;
}
