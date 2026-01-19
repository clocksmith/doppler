

export function validateManifest(manifest) {
  const errors = [];

  // Version check
  const version = typeof manifest.version === 'string'
    ? parseFloat(manifest.version)
    : manifest.version;
  if (typeof version !== 'number' || isNaN(version) || version < 1) {
    errors.push(`Invalid version: ${manifest.version}`);
  }

  // Required string fields
  if (!manifest.modelId || typeof manifest.modelId !== 'string') {
    errors.push('Missing or invalid modelId field');
  }

  if (!manifest.modelType || typeof manifest.modelType !== 'string') {
    errors.push('Missing or invalid modelType field');
  }

  if (!manifest.quantization || typeof manifest.quantization !== 'string') {
    errors.push('Missing or invalid quantization field');
  }

  // Inference config (manifest-first: required for all models)
  if (!manifest.inference || typeof manifest.inference !== 'object') {
    errors.push('Missing or invalid inference field');
  }

  // Hash algorithm (required)
  if (!manifest.hashAlgorithm) {
    errors.push('Missing or invalid hashAlgorithm');
  } else if (manifest.hashAlgorithm !== 'sha256' && manifest.hashAlgorithm !== 'blake3') {
    errors.push(`Invalid hashAlgorithm: ${manifest.hashAlgorithm}`);
  }

  // EOS token ID (required)
  const eosTokenId = manifest.eos_token_id;
  if (eosTokenId === undefined) {
    errors.push('Missing eos_token_id');
  } else if (Array.isArray(eosTokenId)) {
    if (eosTokenId.length === 0 || eosTokenId.some((id) => typeof id !== 'number')) {
      errors.push('Invalid eos_token_id array');
    }
  } else if (typeof eosTokenId !== 'number') {
    errors.push('Invalid eos_token_id');
  }

  // Architecture validation (skip for LoRA adapters)
  const isLoRAAdapter = manifest.adapterType === 'lora' || manifest.modelType === 'lora' || !!manifest.loraConfig;

  if (!isLoRAAdapter && manifest.architecture && typeof manifest.architecture === 'object') {
    const arch = manifest.architecture;
    const requiredFields = [
      'numLayers',
      'hiddenSize',
      'intermediateSize',
      'numAttentionHeads',
      'numKeyValueHeads',
      'headDim',
      'vocabSize',
      'maxSeqLen',
    ];
    for (const field of requiredFields) {
      const value = arch[field];
      if (typeof value !== 'number' || value <= 0) {
        errors.push(`Invalid architecture.${field}`);
      }
    }
  } else if (!isLoRAAdapter && !manifest.architecture) {
    errors.push('Missing architecture field');
  }

  // Groups validation
  if (manifest.groups && typeof manifest.groups === 'object') {
    const numShards = manifest.shards?.length ?? 0;
    for (const [groupId, group] of Object.entries(manifest.groups)) {
      if (!group.type) {
        errors.push(`Group '${groupId}' missing type`);
      }
      if (!group.version || typeof group.version !== 'string') {
        errors.push(`Group '${groupId}' missing or invalid version`);
      }
      if (!Array.isArray(group.shards)) {
        errors.push(`Group '${groupId}' missing shards array`);
      } else {
        for (const shardIdx of group.shards) {
          if (shardIdx < 0 || shardIdx >= numShards) {
            errors.push(`Group '${groupId}' references invalid shard index ${shardIdx}`);
          }
        }
      }
      if (!Array.isArray(group.tensors)) {
        errors.push(`Group '${groupId}' missing tensors array`);
      }
      if (!group.hash || typeof group.hash !== 'string') {
        errors.push(`Group '${groupId}' missing or invalid hash`);
      }
    }
  }

  // Tensors requirement
  const hasTensorsFile = manifest.tensorsFile && typeof manifest.tensorsFile === 'string';
  const hasInlineTensors = manifest.tensors && typeof manifest.tensors === 'object';
  if (!hasTensorsFile && !hasInlineTensors && !manifest.groups) {
    errors.push('Missing tensorsFile and tensors - one is required');
  }
  if (hasInlineTensors) {
    for (const [name, tensor] of Object.entries(manifest.tensors)) {
      if (!tensor.role || typeof tensor.role !== 'string') {
        errors.push(`Tensor "${name}" missing role`);
      }
    }
  }

  // MoE config validation
  if (manifest.moeConfig !== null && manifest.moeConfig !== undefined) {
    const moe = manifest.moeConfig;
    if (typeof moe.numExperts !== 'number' || moe.numExperts <= 0) {
      errors.push('Invalid moeConfig.numExperts');
    }
    if (typeof moe.numExpertsPerToken !== 'number' || moe.numExpertsPerToken <= 0) {
      errors.push('Invalid moeConfig.numExpertsPerToken');
    }
    if (typeof moe.expertFormat !== 'string') {
      errors.push('Invalid moeConfig.expertFormat');
    } else if (moe.expertFormat !== 'mixtral' && moe.expertFormat !== 'gpt-oss') {
      errors.push(`Invalid moeConfig.expertFormat: ${moe.expertFormat}`);
    }
    if (moe.numExpertsPerToken > moe.numExperts) {
      errors.push('numExpertsPerToken cannot exceed numExperts');
    }
  }

  // Shards validation
  if (!Array.isArray(manifest.shards) || manifest.shards.length === 0) {
    errors.push('Missing or empty shards array');
  } else {
    let expectedOffset = 0;
    for (let i = 0; i < manifest.shards.length; i++) {
      const shard = manifest.shards[i];

      if (shard.index !== i) {
        errors.push(`Shard ${i} has incorrect index: ${shard.index}`);
      }

      if (typeof shard.size !== 'number' || shard.size <= 0) {
        errors.push(`Shard ${i} has invalid size`);
      }

      const hash = shard.hash;
      if (!hash || typeof hash !== 'string' || hash.length !== 64) {
        errors.push(`Shard ${i} has invalid hash`);
      }

      if (!shard.filename || typeof shard.filename !== 'string') {
        errors.push(`Shard ${i} has invalid filename`);
      }

      if (shard.offset !== expectedOffset) {
        errors.push(`Shard ${i} has incorrect offset: expected ${expectedOffset}, got ${shard.offset}`);
      }
      expectedOffset += shard.size;
    }

    if (manifest.totalSize !== expectedOffset) {
      errors.push(`totalSize mismatch: declared ${manifest.totalSize}, calculated ${expectedOffset}`);
    }
  }

  return { valid: errors.length === 0, errors };
}
