import { WgslReflect, WgslScanner } from 'wgsl_reflect/wgsl_reflect.module.js';

// Reflect declarations, never transform the executable shader. The reflection
// parser does not support every valid expression in our function bodies. Tokens
// preserve declarations and balanced bodies without interpreting those expressions.
export function projectWgslInterface(source) {
  const tokens = new WgslScanner(source).scanTokens();
  const chunks = [];
  let start = 0;
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].lexeme !== 'fn') continue;
    while (i < tokens.length && tokens[i].lexeme !== '{') i++;
    if (i === tokens.length) throw new Error('WGSL function has no body.');
    chunks.push(source.slice(start, tokens[i].end));
    let depth = 1;
    i++;
    while (i < tokens.length && depth > 0) {
      if (tokens[i].lexeme === '{') depth++;
      else if (tokens[i].lexeme === '}') depth--;
      if (depth > 0) i++;
    }
    if (depth !== 0) throw new Error('WGSL function body is unbalanced.');
    start = tokens[i].start;
  }
  chunks.push(source.slice(start));
  return chunks.join('');
}

export function reflectWgslInterface(source) {
  const reflection = new WgslReflect(projectWgslInterface(source));
  const tokens = new WgslScanner(source).scanTokens();
  return {
    memberReferences: [...new Set(tokens.filter((token, index) => tokens[index - 1]?.lexeme === '.')
      .map((token) => token.lexeme))],
    uniforms: reflection.uniforms.map((uniform) => ({
      name: uniform.name,
      structName: uniform.type.name,
      group: uniform.group,
      binding: uniform.binding,
      size: uniform.size,
      fields: uniform.members.map((field) => ({
        name: field.name,
        type: field.type.format ? `${field.type.name}<${field.type.format.name}>` : field.type.name,
        offset: field.offset,
        size: field.size,
      })),
    })),
    bindings: [...reflection.uniforms, ...reflection.storage].map((binding) => ({
      name: binding.name,
      group: binding.group,
      index: binding.binding,
      type: reflection.uniforms.includes(binding) ? 'uniform'
        : binding.access === 'read' ? 'read-only-storage' : 'storage',
    })).sort((a, b) => a.group - b.group || a.index - b.index),
    entryPoints: reflection.entry.compute.map((entry) => entry.name),
  };
}

export function compareUniformInterface(uniforms, reflected, label) {
  const errors = [];
  if (reflected.length === 0) {
    if (uniforms != null) errors.push(`${label}: registry declares uniforms absent from WGSL.`);
    return errors;
  }
  if (reflected.length !== 1) {
    return [`${label}: registry supports one uniform buffer per variant.`];
  }
  const actual = reflected[0];
  if (!uniforms) return [`${label}: registry is missing the ${actual.size}-byte uniform layout.`];
  if (!Array.isArray(uniforms.fields)) return [`${label}: registry uniform fields must be an array.`];
  if (uniforms.size !== actual.size) {
    errors.push(`${label}: uniform size ${uniforms.size}; WGSL requires ${actual.size}.`);
  }
  for (const field of actual.fields) {
    const declared = uniforms.fields.find((candidate) => candidate.name === field.name);
    if (!declared || declared.type !== field.type || declared.offset !== field.offset) {
      errors.push(`${label}: ${field.name} must be ${field.type} at byte ${field.offset}.`);
    }
  }
  for (const field of uniforms.fields) {
    if (!actual.fields.some((candidate) => candidate.name === field.name)) {
      errors.push(`${label}: ${field.name} is absent from the WGSL uniform struct.`);
    }
  }
  return errors;
}
