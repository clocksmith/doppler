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
  throw new Error(
    `sourceTensors.rules[${index}].kind must be "rename", "split", or "ignore".`
  );
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
  }
  assertUniqueTargetNames(outputs);
  return outputs;
}
