// Lexical WGSL style checks, not a shader compiler or a semantic proof.
// Inventory, exceptions, and CLI reporting belong to the shared style contract.

export const RULE_IDS = Object.freeze([
  'wgsl-comments',
  'wgsl-override-name',
  'wgsl-function-name',
  'wgsl-padding-access',
  'wgsl-f16-enable',
  'wgsl-binding-unique',
]);

function maskComments(source) {
  let depth = 0;
  let lineComment = false;
  const chars = source.split('');
  for (let index = 0; index < chars.length; index++) {
    if (lineComment && chars[index] === '\n') lineComment = false;
    if (!lineComment && source.startsWith('/*', index)) {
      depth++;
      chars[index++] = ' ';
      chars[index] = ' ';
    } else if (depth > 0 && source.startsWith('*/', index)) {
      depth--;
      chars[index++] = ' ';
      chars[index] = ' ';
    } else if (!depth && source.startsWith('//', index)) {
      lineComment = true;
      chars[index] = ' ';
    } else if ((depth || lineComment) && chars[index] !== '\n') {
      chars[index] = ' ';
    }
  }
  return { code: chars.join(''), depth };
}

export function checkWgslSource(source, path) {
  const { code, depth } = maskComments(source);
  const diagnostics = [];
  function report(ruleId, message, index) {
    diagnostics.push({ ruleId, message, line: code.slice(0, index).split('\n').length });
  }
  if (depth) report('wgsl-comments', 'Unterminated block comment.', code.length);
  for (const match of code.matchAll(/\b(override|fn)\s+([A-Za-z_][A-Za-z_0-9]*)/g)) {
    const [, kind, name] = match;
    const valid = kind === 'override' ? /^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$/ : /^_?[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/;
    if (!valid.test(name)) report(`wgsl-${kind === 'fn' ? 'function' : 'override'}-name`, `${kind} ${name} must use ${kind === 'fn' ? 'snake_case' : 'UPPER_SNAKE_CASE'}.`, match.index);
  }
  for (const match of code.matchAll(/\.\s*(_pad\w*)\b/g)) {
    report('wgsl-padding-access', `${match[1]} is padding, not runtime state; declare an explicit uniform field.`, match.index);
  }
  // Include fragments inherit enable directives from the assembled module.
  if (!path.endsWith('.wgslinc') && /\bf16\b|\b(?:vec[234]|mat[234]x[234])h\b/.test(code)
    && !/\benable\s+[^;]*\bf16\b[^;]*;/.test(code)) {
    report('wgsl-f16-enable', 'f16 use requires an explicit enable f16 directive.', 0);
  }
  const bindings = new Set();
  for (const match of code.matchAll(/((?:@\w+\s*\([^)]*\)\s*)+)var\s*</g)) {
    const group = /@group\s*\(\s*(\d+)u?\s*\)/.exec(match[1]);
    const binding = /@binding\s*\(\s*(\d+)u?\s*\)/.exec(match[1]);
    if (!group || !binding) continue; // Symbolic attributes remain compiler-checked.
    const key = `${Number(group[1])}:${Number(binding[1])}`;
    if (bindings.has(key)) report('wgsl-binding-unique', `Duplicate numeric group:binding ${key}.`, match.index);
    bindings.add(key);
  }
  return diagnostics;
}
