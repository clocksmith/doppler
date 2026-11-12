// A WGSL `enable` directive applies to the whole module. Several kernels keep a
// subgroup entry point beside plain ones in one file and declare
// `enable subgroups;` at the top. A device without the subgroups feature then
// rejects the whole module, including the entry points that never touch a
// subgroup builtin, before variant selection has any say.
//
// This filter gives such a device the module it can compile: the directive is
// removed together with every function that uses subgroup builtins, directly
// or through a function it calls. Devices that offer subgroups receive the
// source unchanged. A module left with no compute entry point needs the feature
// outright, and that is reported as such instead of as a compiler failure.

const ENABLE_SUBGROUPS = /^[ \t]*enable[ \t]+subgroups[ \t]*;[ \t]*\r?\n?/m;
// The compute stage attribute, written escaped: this file matches shader text and holds none.
const COMPUTE_STAGE = /\x40compute\b/;
const SUBGROUP_USE = /\bsubgroup[A-Z]\w*\s*\(|@builtin\(\s*subgroup_\w+\s*\)/;

// Comments are blanked, not removed, so offsets into the stripped text are
// offsets into the source.
/** @param {string} source */
function blankComments(source) {
  let out = '';
  let index = 0;
  while (index < source.length) {
    if (source.startsWith('//', index)) {
      const end = source.indexOf('\n', index);
      const stop = end === -1 ? source.length : end;
      out += ' '.repeat(stop - index);
      index = stop;
    } else if (source.startsWith('/*', index)) {
      let depth = 1;
      let cursor = index + 2;
      while (cursor < source.length && depth > 0) {
        if (source.startsWith('/*', cursor)) { depth += 1; cursor += 2; }
        else if (source.startsWith('*/', cursor)) { depth -= 1; cursor += 2; }
        else cursor += 1;
      }
      out += source.slice(index, cursor).replace(/[^\n]/g, ' ');
      index = cursor;
    } else {
      out += source[index];
      index += 1;
    }
  }
  return out;
}

// Every module-scope function: its name, the span from its first attribute to
// its closing brace, and its text with comments blanked.
/** @param {string} source */
function moduleFunctions(source) {
  const code = blankComments(source);
  /** @type {{ name: string, start: number, end: number, text: string }[]} */
  const functions = [];
  const header = /(^|\n)((?:[ \t]*@[^\n]*\n)*[ \t]*(?:@[^\n{]*?\s)?fn\s+([A-Za-z_]\w*)\s*\()/g;
  let match;
  while ((match = header.exec(code)) !== null) {
    const start = match.index + match[1].length;
    const open = code.indexOf('{', match.index + match[0].length);
    if (open === -1) break;
    let depth = 0;
    let end = -1;
    for (let cursor = open; cursor < code.length; cursor += 1) {
      if (code[cursor] === '{') depth += 1;
      else if (code[cursor] === '}') { depth -= 1; if (depth === 0) { end = cursor + 1; break; } }
    }
    if (end === -1) break;
    functions.push({ name: match[3], start, end, text: code.slice(start, end) });
    header.lastIndex = end;
  }
  return functions;
}

/** @param {string} source */
export function shaderEnablesSubgroups(source) {
  return ENABLE_SUBGROUPS.test(blankComments(String(source ?? '')));
}

// The source a device can compile, whether it was filtered, and the functions removed.
/**
 * @param {string} source
 * @param {{ hasSubgroups: boolean, label?: string }} capabilities
 */
export function filterShaderForCapabilities(source, { hasSubgroups, label = 'shader' }) {
  const text = String(source ?? '');
  if (hasSubgroups || !shaderEnablesSubgroups(text)) {
    return { source: text, filtered: false, removedFunctions: [] };
  }
  const functions = moduleFunctions(text);
  const removed = new Set(functions.filter((fn) => SUBGROUP_USE.test(fn.text)).map((fn) => fn.name));
  // A function that calls a removed function goes too.
  for (let changed = true; changed;) {
    changed = false;
    for (const fn of functions) {
      if (removed.has(fn.name)) continue;
      if ([...removed].some((name) => new RegExp(`\\b${name}\\s*\\(`).test(fn.text.slice(fn.text.indexOf('{'))))) {
        removed.add(fn.name);
        changed = true;
      }
    }
  }
  let filtered = '';
  let cursor = 0;
  for (const fn of functions) {
    if (!removed.has(fn.name)) continue;
    filtered += text.slice(cursor, fn.start);
    cursor = fn.end;
  }
  filtered += text.slice(cursor);
  const directive = blankComments(filtered).match(ENABLE_SUBGROUPS);
  if (directive) {
    const at = directive.index ?? 0;
    filtered = filtered.slice(0, at) + filtered.slice(at + directive[0].length);
  }
  const remaining = blankComments(filtered);
  if (SUBGROUP_USE.test(remaining) || !COMPUTE_STAGE.test(remaining)) {
    throw new Error(`${label} requires the WebGPU "subgroups" feature, which this device does not offer.`);
  }
  return { source: filtered, filtered: true, removedFunctions: [...removed] };
}
