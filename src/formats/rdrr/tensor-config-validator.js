// Tensor-Config Consistency Validator
//
// Validates that manifest config flags are consistent with the actual tensors present.
// This catches bugs like postFeedforwardNorm=false when the weights exist.
// See tensor-config-validator.d.ts for type definitions.

// Tensor name patterns for each config flag
const TENSOR_CONFIG_MAPPINGS = [
  {
    configPath: 'inference.normalization.postFeedforwardNorm',
    tensorPattern: /post_feedforward_layernorm\.weight$/,
    description: 'post-feedforward normalization',
    severity: 'error', // This caused complete model failure
  },
  {
    configPath: 'inference.normalization.preFeedforwardNorm',
    tensorPattern: /pre_feedforward_layernorm\.weight$/,
    description: 'pre-feedforward normalization',
    severity: 'error',
  },
  {
    configPath: 'inference.normalization.postAttentionNorm',
    tensorPattern: /post_attention_layernorm\.weight$/,
    description: 'post-attention normalization',
    severity: 'error',
  },
  {
    configPath: 'inference.attention.queryKeyNorm',
    tensorPattern: /self_attn\.(q_norm|k_norm)\.weight$/,
    description: 'query/key normalization',
    severity: 'warning',
  },
  {
    configPath: 'inference.output.tieWordEmbeddings',
    tensorPattern: /^(lm_head\.weight|model\.lm_head\.weight)$/,
    description: 'separate LM head weights',
    invertLogic: true, // If tensor exists, tieWordEmbeddings should be FALSE
    severity: 'warning',
  },
];

// Get nested property from object using dot notation.
function getNestedValue(obj, path) {
  return path.split('.').reduce((current, key) => current?.[key], obj);
}

// Collect all tensor names from manifest.
function collectTensorNames(manifest) {
  const tensorNames = [];

  // From groups
  if (manifest.groups) {
    for (const group of Object.values(manifest.groups)) {
      if (Array.isArray(group.tensors)) {
        tensorNames.push(...group.tensors);
      }
    }
  }

  // From inline tensors
  if (manifest.tensors) {
    tensorNames.push(...Object.keys(manifest.tensors));
  }

  return tensorNames;
}

// Validate tensor-config consistency.
export function validateTensorConfigConsistency(manifest) {
  const warnings = [];
  const errors = [];

  const tensorNames = collectTensorNames(manifest);

  for (const mapping of TENSOR_CONFIG_MAPPINGS) {
    const configValue = getNestedValue(manifest, mapping.configPath);
    const matchingTensors = tensorNames.filter(name => mapping.tensorPattern.test(name));
    const hasTensors = matchingTensors.length > 0;

    // Check for inconsistency
    let isInconsistent = false;
    let suggestion = '';

    if (mapping.invertLogic) {
      // For inverted logic (e.g., tieWordEmbeddings)
      // If tensors exist, config should be false
      if (hasTensors && configValue === true) {
        isInconsistent = true;
        suggestion = `Set ${mapping.configPath} to false (found ${matchingTensors.length} tensor(s): ${matchingTensors.slice(0, 2).join(', ')}...)`;
      }
    } else {
      // Normal logic: if tensors exist, config should be true
      if (hasTensors && configValue === false) {
        isInconsistent = true;
        suggestion = `Set ${mapping.configPath} to true (found ${matchingTensors.length} tensor(s): ${matchingTensors.slice(0, 2).join(', ')}...)`;
      }
      // Also warn if config is true but no tensors found
      if (!hasTensors && configValue === true) {
        isInconsistent = true;
        suggestion = `Set ${mapping.configPath} to false (no matching tensors found for pattern: ${mapping.tensorPattern})`;
      }
    }

    if (isInconsistent) {
      const issue = {
        severity: mapping.severity,
        code: `TENSOR_CONFIG_MISMATCH_${mapping.configPath.replace(/\./g, '_').toUpperCase()}`,
        message: `Config flag "${mapping.configPath}" (=${configValue}) is inconsistent with ${mapping.description} tensors (found=${hasTensors})`,
        suggestion,
      };

      if (mapping.severity === 'error') {
        errors.push(issue);
      } else {
        warnings.push(issue);
      }
    }
  }

  return {
    valid: errors.length === 0,
    warnings,
    errors,
  };
}

// Format validation result for logging.
export function formatValidationResult(result) {
  const lines = [];

  if (result.errors.length > 0) {
    lines.push('ERRORS:');
    for (const err of result.errors) {
      lines.push(`  [${err.code}] ${err.message}`);
      if (err.suggestion) {
        lines.push(`    -> ${err.suggestion}`);
      }
    }
  }

  if (result.warnings.length > 0) {
    lines.push('WARNINGS:');
    for (const warn of result.warnings) {
      lines.push(`  [${warn.code}] ${warn.message}`);
      if (warn.suggestion) {
        lines.push(`    -> ${warn.suggestion}`);
      }
    }
  }

  if (result.valid && result.warnings.length === 0) {
    lines.push('Tensor-config consistency: OK');
  }

  return lines.join('\n');
}
