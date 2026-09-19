export function getRequiredWgslFeatures(source) {
  if (typeof source !== 'string') throw new Error('WGSL language contract requires source text.');
  const chunks = [];
  let start = 0;
  for (let index = 0; index < source.length; index++) {
    if (source[index] !== '/' || !['/', '*'].includes(source[index + 1])) continue;
    chunks.push(source.slice(start, index), ' ');
    if (source[index + 1] === '/') {
      index = source.indexOf('\n', index + 2);
      if (index < 0) { start = source.length; break; }
    } else {
      let depth = 1;
      index += 2;
      while (index < source.length && depth > 0) {
        if (source.startsWith('/*', index)) { depth++; index += 2; }
        else if (source.startsWith('*/', index)) { depth--; index += 2; }
        else index++;
      }
      if (depth !== 0) throw new Error('WGSL source has an unterminated block comment.');
      index--;
    }
    start = index + 1;
  }
  chunks.push(source.slice(start));
  const code = chunks.join('');
  const features = new Set();
  for (const match of code.matchAll(/\brequires\b\s*([^;]*);/g)) {
    const items = match[1].split(',');
    if (items.at(-1).trim() === '') items.pop();
    if (items.length === 0) throw new Error('WGSL requires directive must name a language feature.');
    for (const item of items) {
      const feature = item.trim();
      if (!/^[a-z][a-z0-9_]*$/.test(feature)) throw new Error(`Invalid WGSL language feature "${feature}".`);
      features.add(feature);
    }
  }
  // WGSL permits use of supported language extensions without a requires
  // directive. The signed execution recipe must still declare these built-ins.
  if (/@builtin\s*\(\s*(?:subgroup_id|num_subgroups)\s*\)/.test(code)) features.add('subgroup_id');
  return [...features].sort();
}

export function assertWgslFeaturesSupported(required, available, label) {
  const supported = new Set(available ?? []);
  const missing = required.filter(feature => !supported.has(feature));
  if (missing.length) throw new Error(`${label} requires unsupported WGSL language features: ${missing.join(', ')}.`);
}
