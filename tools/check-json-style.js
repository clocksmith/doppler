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

export function checkJsonSource(source, filePath = '') {
  const text = jsonText(source, filePath);
  try {
    JSON.parse(text);
  } catch (error) {
    return [{ ruleId: 'JSON-001', message: `Invalid JSON: ${error.message}` }];
  }
  const issues = [];
  const stack = [];
  // JSON.parse establishes grammar first. This token walk preserves members that
  // JSON.parse would silently overwrite, including escaped equivalent key names.
  const tokens = /"(?:\\[\s\S]|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[{}\[\]:,]|true|false|null/g;
  let line = 1;
  let lastIndex = 0;
  for (const match of text.matchAll(tokens)) {
    const token = match[0];
    for (let index = lastIndex; index < match.index; index += 1) {
      if (text[index] === '\n') line += 1;
    }
    lastIndex = match.index;
    if (token === '{' || token === '[') {
      stack.push(token === '{' ? new Set() : null);
    } else if (token === '}' || token === ']') {
      stack.pop();
    } else if (token.startsWith('"') && /^\s*:/.test(text.slice(match.index + token.length))) {
      const keys = stack[stack.length - 1];
      const key = JSON.parse(token);
      if (keys.has(key)) {
        issues.push({ ruleId: 'JSON-002', line, message: `Duplicate object member ${JSON.stringify(key)}; JSON.parse silently overwrites it.` });
      }
      keys.add(key);
    } else if (/^-?\d/.test(token) && !Number.isFinite(Number(token))) {
      issues.push({ ruleId: 'JSON-003', line, message: `Number ${token} overflows JavaScript's finite numeric range.` });
    }
  }
  return issues;
}
