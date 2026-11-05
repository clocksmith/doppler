function requirePlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function requireString(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value;
}

function compileRulePattern(rule, index) {
  const source = requireString(rule.match, `sourceTensors.rules[${index}].match`);
  try {
    return new RegExp(source);
  } catch (error) {
    throw new Error(
      `sourceTensors.rules[${index}].match is not a valid regular expression: ${error.message}`
    );
  }
}

function requireExpectedMatches(rule, index) {
  const value = rule.expectedMatches;
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(
      `sourceTensors.rules[${index}].expectedMatches must be a non-negative integer.`
    );
  }
  return value;
}

function applyDescriptorOverrides(tensor, rule) {
  return {
    ...tensor,
    ...(rule.role ? { role: requireString(rule.role, 'source tensor role') } : {}),
    ...(rule.group ? { group: requireString(rule.group, 'source tensor group') } : {}),
  };
}

function renameTensor(tensor, rule, pattern, index) {
  const replacement = requireString(rule.replace, `sourceTensors.rules[${index}].replace`);
  const name = tensor.name.replace(pattern, replacement);
  if (!name || name === tensor.name && !pattern.test(tensor.name)) {
    throw new Error(`sourceTensors.rules[${index}] did not produce a target tensor name.`);
  }
  return [applyDescriptorOverrides({ ...tensor, name }, rule)];
}

function splitTensor(tensor, rule, pattern, index) {
  if (rule.axis !== 0) {
    throw new Error(`sourceTensors.rules[${index}].axis must be 0 for contiguous SafeTensors splits.`);
  }
  if (!Array.isArray(tensor.shape) || tensor.shape.length < 1) {
    throw new Error(`Cannot split source tensor "${tensor.name}" without a shape.`);
  }
  if (!Array.isArray(rule.parts) || rule.parts.length < 2) {
    throw new Error(`sourceTensors.rules[${index}].parts must contain at least two parts.`);
  }
  const leadingSize = Number(tensor.shape[0]);
  const rowByteSize = tensor.size / leadingSize;
  if (!Number.isInteger(leadingSize) || leadingSize <= 0 || !Number.isInteger(rowByteSize)) {
    throw new Error(`Source tensor "${tensor.name}" cannot be split into contiguous rows.`);
  }

  let rowOffset = 0;
  const outputs = rule.parts.map((rawPart, partIndex) => {
    const part = requirePlainObject(
      rawPart,
      `sourceTensors.rules[${index}].parts[${partIndex}]`
    );
    const rows = part.size;
    if (!Number.isInteger(rows) || rows <= 0) {
      throw new Error(
        `sourceTensors.rules[${index}].parts[${partIndex}].size must be a positive integer.`
      );
    }
    const replacement = requireString(
      part.replace,
      `sourceTensors.rules[${index}].parts[${partIndex}].replace`
    );
    const output = applyDescriptorOverrides({
      ...tensor,
      name: tensor.name.replace(pattern, replacement),
      shape: [rows, ...tensor.shape.slice(1)],
      offset: tensor.offset + rowOffset * rowByteSize,
      size: rows * rowByteSize,
    }, part);
    rowOffset += rows;
    return output;
  });

  if (rowOffset !== leadingSize) {
    throw new Error(
      `sourceTensors.rules[${index}] split ${rowOffset} rows from "${tensor.name}", expected ${leadingSize}.`
    );
  }
  return outputs;
}

function applyRule(tensor, compiledRule, index) {
  const { rule, pattern } = compiledRule;
  if (rule.kind === 'rename') {
    return renameTensor(tensor, rule, pattern, index);
  }
  if (rule.kind === 'split') {
    return splitTensor(tensor, rule, pattern, index);
  }
  if (rule.kind === 'ignore') {
    requireString(rule.reason, `sourceTensors.rules[${index}].reason`);
    return [];
  }
  if (rule.kind === 'retain') {
    return [retainTensor(tensor, compiledRule, index)];
  }
  throw new Error(
    `sourceTensors.rules[${index}].kind must be "rename", "split", "ignore", or "retain".`
  );
}

// A retained tensor keeps its source bytes: conversion links the pinned source
// file into the artifact as one shard and records the tensor's byte span there
// (src/converter/retained-source-shards.js). Each source file is pinned by the
// SHA-256 of its complete bytes.
function retainTensor(tensor, compiledRule, index) {
  const { rule } = compiledRule;
  requireString(rule.reason, `sourceTensors.rules[${index}].reason`);
  const role = requireString(rule.role, `sourceTensors.rules[${index}].role`);
  // An explicit group keeps retained tensors out of per-layer shard prefetch.
  const group = requireString(rule.group, `sourceTensors.rules[${index}].group`);
  const files = requirePlainObject(rule.files, `sourceTensors.rules[${index}].files`);
  const sourcePath = requireString(tensor.sourcePath, `retained source tensor "${tensor.name}" sourcePath`);
  const file = sourcePath.split(/[\\/]/).pop();
  const sha256 = Object.hasOwn(files, file) ? files[file] : undefined;
  if (typeof sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(sha256)) {
    throw new Error(`sourceTensors.rules[${index}].files lacks a SHA-256 pin for "${file}" (tensor "${tensor.name}").`);
  }
  compiledRule.retainedFiles.add(file);
  return { ...tensor, role, group, retainedSource: { file, sha256 } };
}

function assertUniqueTargetNames(tensors) {
  const names = new Set();
  for (const tensor of tensors) {
    if (names.has(tensor.name)) {
      throw new Error(`sourceTensors rules produced duplicate target tensor "${tensor.name}".`);
    }
    names.add(tensor.name);
  }
}

export function applySourceTensorRules(tensors, policy) {
  if (policy == null) return tensors;
  requirePlainObject(policy, 'sourceTensors');
  if (typeof policy.requireAll !== 'boolean') {
    throw new Error('sourceTensors.requireAll must be boolean.');
  }
  if (!Array.isArray(policy.rules) || policy.rules.length === 0) {
    throw new Error('sourceTensors.rules must be a non-empty array.');
  }
  const compiledRules = policy.rules.map((rawRule, index) => {
    const rule = requirePlainObject(rawRule, `sourceTensors.rules[${index}]`);
    return {
      rule,
      pattern: compileRulePattern(rule, index),
      expectedMatches: requireExpectedMatches(rule, index),
      matches: 0,
      retainedFiles: new Set(),
    };
  });

  const outputs = [];
  for (const tensor of tensors) {
    const matches = compiledRules
      .map((entry, index) => ({ entry, index }))
      .filter(({ entry }) => entry.pattern.test(tensor.name));
    if (matches.length > 1) {
      throw new Error(`Source tensor "${tensor.name}" matches multiple sourceTensors rules.`);
    }
    if (matches.length === 0) {
      if (policy.requireAll) {
        throw new Error(`Source tensor "${tensor.name}" is not covered by sourceTensors rules.`);
      }
      outputs.push(tensor);
      continue;
    }
    const [{ entry, index }] = matches;
    entry.matches += 1;
    outputs.push(...applyRule(tensor, entry, index));
  }

  for (let index = 0; index < compiledRules.length; index += 1) {
    const entry = compiledRules[index];
    if (entry.matches !== entry.expectedMatches) {
      throw new Error(
        `sourceTensors.rules[${index}] matched ${entry.matches} tensors, expected ${entry.expectedMatches}.`
      );
    }
    const unused = entry.rule.kind === 'retain'
      ? Object.keys(entry.rule.files).filter((file) => !entry.retainedFiles.has(file)) : [];
    if (unused.length > 0) {
      throw new Error(`sourceTensors.rules[${index}].files pins ${unused.join(', ')} but no matched tensor is stored there.`);
    }
  }
  assertUniqueTargetNames(outputs);
  return outputs;
}
