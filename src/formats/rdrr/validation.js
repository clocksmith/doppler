import { validateTensorConfigConsistency } from './tensor-config-validator.js';
import { validateManifestExecutionContract } from '../../config/execution-contract-check.js';
import { validateManifestInference } from '../../config/schema/index.js';
import { validateTensorStorageDescriptor } from './storage-descriptor.js';

const ARTIFACT_COMPLETENESS_VALUES = new Set([
  'complete',
  'weights-ref',
  'incomplete',
]);

const LOWERING_EXACTNESS_CLASSES = new Set([
  'bit_exact_solo',
  'algorithm_exact',
  'tolerance_bounded',
]);

const LOWERING_NULLABLE_STRING_FIELDS = [
  'targetDescriptorCorrectnessHash',
  'frontendVersion',
  'tsirSemanticDigest',
  'tsirRealizationDigest',
  'emitterDigest',
  'compilerVersion',
];

const LOWERING_ALGORITHM_EXACT_INVARIANTS = new Set([
  'reduction_order',
  'tree_shape',
  'accum_dtype',
  'associativity_grouping',
]);

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function validateOptionalNonEmptyString(errors, object, field, prefix) {
  if (object[field] === undefined) {
    return;
  }
  if (typeof object[field] !== 'string' || object[field].trim().length === 0) {
    errors.push(`Invalid ${prefix}.${field}`);
  }
}

function validateRequiredNonEmptyString(errors, object, field, prefix) {
  if (typeof object[field] !== 'string' || object[field].trim().length === 0) {
    errors.push(`Missing or invalid ${prefix}.${field}`);
  }
}

function validateIntegrityExtensionsContract(manifest, errors) {
  const integrityExtensions = manifest.integrityExtensions;
  if (integrityExtensions === undefined) {
    return;
  }
  if (!isPlainObject(integrityExtensions)) {
    errors.push('Invalid integrityExtensions field');
    return;
  }
  if (integrityExtensions.contractVersion !== 1) {
    errors.push(`Invalid integrityExtensions.contractVersion: ${integrityExtensions.contractVersion}`);
  }
  if (!isPlainObject(integrityExtensions.blockMerkle)) {
    errors.push('Invalid integrityExtensions.blockMerkle');
    return;
  }
  const blockSize = integrityExtensions.blockMerkle.blockSize;
  if (!Number.isInteger(blockSize) || blockSize <= 0) {
    errors.push('Invalid integrityExtensions.blockMerkle.blockSize');
  }
  const roots = integrityExtensions.blockMerkle.roots;
  if (!isPlainObject(roots) || Object.keys(roots).length === 0) {
    errors.push('Invalid integrityExtensions.blockMerkle.roots');
    return;
  }
  for (const [tensorId, merkleRoot] of Object.entries(roots)) {
    if (typeof tensorId !== 'string' || tensorId.trim().length === 0) {
      errors.push('Invalid integrityExtensions.blockMerkle.roots tensor id');
      continue;
    }
    if (typeof merkleRoot !== 'string' || merkleRoot.trim().length === 0) {
      errors.push(`Invalid integrityExtensions.blockMerkle.roots.${tensorId}`);
    }
  }

  validateLoweringsSection(integrityExtensions.lowerings, errors);
}

function validateLoweringsSection(lowerings, errors) {
  if (lowerings === undefined) {
    return;
  }
  if (!isPlainObject(lowerings)) {
    errors.push('Invalid integrityExtensions.lowerings');
    return;
  }
  if (lowerings.contractVersion !== 1) {
    errors.push(`Invalid integrityExtensions.lowerings.contractVersion: ${lowerings.contractVersion}`);
  }
  if (!Array.isArray(lowerings.entries)) {
    errors.push('Invalid integrityExtensions.lowerings.entries');
    return;
  }
  const seen = new Set();
  lowerings.entries.forEach((entry, index) => {
    const prefix = `integrityExtensions.lowerings.entries[${index}]`;
    if (!isPlainObject(entry)) {
      errors.push(`${prefix} must be an object`);
      return;
    }
    if (typeof entry.kernelRef !== 'string' || entry.kernelRef.trim().length === 0) {
      errors.push(`${prefix}.kernelRef must be a non-empty string`);
    }
    if (typeof entry.backend !== 'string' || entry.backend.trim().length === 0) {
      errors.push(`${prefix}.backend must be a non-empty string`);
    }
    if (typeof entry.kernelRef === 'string' && typeof entry.backend === 'string') {
      const key = `${entry.kernelRef}::${entry.backend}`;
      if (seen.has(key)) {
        errors.push(`${prefix} duplicates (kernelRef, backend) pair already declared earlier`);
      }
      seen.add(key);
    }
    for (const field of LOWERING_NULLABLE_STRING_FIELDS) {
      const value = entry[field];
      if (value === undefined) {
        errors.push(`${prefix}.${field} is required (use null for not-applicable)`);
        continue;
      }
      if (value === null) continue;
      if (typeof value !== 'string' || value.trim().length === 0) {
        errors.push(`${prefix}.${field} must be a non-empty string or null`);
      }
    }
    if (entry.exactness === undefined) {
      errors.push(`${prefix}.exactness is required (use null for rejection entries)`);
    } else if (entry.exactness !== null) {
      if (!isPlainObject(entry.exactness)) {
        errors.push(`${prefix}.exactness must be an object or null`);
      } else {
        if (!LOWERING_EXACTNESS_CLASSES.has(entry.exactness.class)) {
          errors.push(`${prefix}.exactness.class must be one of bit_exact_solo, algorithm_exact, tolerance_bounded`);
        }
        if (!Array.isArray(entry.exactness.algorithmExactInvariants)) {
          errors.push(`${prefix}.exactness.algorithmExactInvariants must be an array`);
        } else {
          entry.exactness.algorithmExactInvariants.forEach((inv, i) => {
            if (typeof inv !== 'string' || !LOWERING_ALGORITHM_EXACT_INVARIANTS.has(inv)) {
              errors.push(`${prefix}.exactness.algorithmExactInvariants[${i}] must be one of reduction_order, tree_shape, accum_dtype, associativity_grouping`);
            }
          });
        }
        if (typeof entry.exactness.toleranceMetric !== 'string') {
          errors.push(`${prefix}.exactness.toleranceMetric must be a string`);
        }
        if (typeof entry.exactness.toleranceEpsilon !== 'number') {
          errors.push(`${prefix}.exactness.toleranceEpsilon must be a number`);
        }
      }
    }
    if (entry.rejectionReasons === undefined) {
      errors.push(`${prefix}.rejectionReasons is required (empty array when backend honored the kernel)`);
    } else if (!Array.isArray(entry.rejectionReasons)) {
      errors.push(`${prefix}.rejectionReasons must be an array (empty for success)`);
    } else {
      entry.rejectionReasons.forEach((reason, reasonIndex) => {
        if (typeof reason !== 'string' || reason.trim().length === 0) {
          errors.push(`${prefix}.rejectionReasons[${reasonIndex}] must be a non-empty string`);
        }
      });
    }

    // A lowering receipt is either a success (digests populated, rejectionReasons empty)
    // or a rejection (digests null, rejectionReasons non-empty). Mixed states are invalid.
    const hasDigests = (
      typeof entry.tsirSemanticDigest === 'string'
      && typeof entry.tsirRealizationDigest === 'string'
      && typeof entry.emitterDigest === 'string'
    );
    const allDigestsNull = (
      entry.tsirSemanticDigest === null
      && entry.tsirRealizationDigest === null
      && entry.emitterDigest === null
    );
    const hasRejection = Array.isArray(entry.rejectionReasons) && entry.rejectionReasons.length > 0;
    if (hasRejection && hasDigests) {
      errors.push(`${prefix} has both rejectionReasons and digests; must be one or the other`);
    }
    if (!hasRejection && !hasDigests && !allDigestsNull) {
      errors.push(`${prefix} has inconsistent digests: all must be strings (success) or all null (rejection)`);
    }
    if (!hasRejection && allDigestsNull && !hasDigests) {
      errors.push(`${prefix} declares no state: success requires populated digests with empty rejectionReasons, rejection requires null digests with non-empty rejectionReasons`);
    }
  });
}

function validateArtifactIdentityContract(manifest, errors) {
  const identity = manifest.artifactIdentity;
  if (identity !== undefined) {
    if (!isPlainObject(identity)) {
      errors.push('Invalid artifactIdentity field');
    } else {
      for (const field of [
        'sourceCheckpointId',
        'sourceRepo',
        'sourceRevision',
        'sourceFormat',
        'conversionConfigPath',
        'conversionConfigDigest',
        'weightPackId',
        'weightPackHash',
        'shardSetHash',
        'manifestVariantId',
        'materializationProfile',
      ]) {
        validateOptionalNonEmptyString(errors, identity, field, 'artifactIdentity');
      }
      if (identity.modalitySet !== undefined) {
        if (!Array.isArray(identity.modalitySet) || identity.modalitySet.some((item) => typeof item !== 'string' || item.trim().length === 0)) {
          errors.push('Invalid artifactIdentity.modalitySet');
        }
      }
      if (identity.artifactCompleteness !== undefined && !ARTIFACT_COMPLETENESS_VALUES.has(identity.artifactCompleteness)) {
        errors.push(`Invalid artifactIdentity.artifactCompleteness: ${identity.artifactCompleteness}`);
      }
    }
  }

  const weightsRef = manifest.weightsRef;
  if (weightsRef !== undefined) {
    if (!isPlainObject(weightsRef)) {
      errors.push('Invalid weightsRef field');
    } else {
      for (const field of ['weightPackId', 'artifactRoot', 'manifestDigest', 'shardSetHash']) {
        validateRequiredNonEmptyString(errors, weightsRef, field, 'weightsRef');
      }
    }
  }
}

export function validateManifest(manifest) {
  const errors = [];
  const warnings = [];

  const isDiffusion = manifest.modelType === 'diffusion';
  const isEnergy = manifest.modelType === 'energy';
  const isEmbedding = manifest.modelType === 'embedding';

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

  validateArtifactIdentityContract(manifest, errors);
  validateIntegrityExtensionsContract(manifest, errors);

  // EOS token ID (required for text models)
  const eosTokenId = manifest.eos_token_id;
  if (!isDiffusion && !isEnergy && !isEmbedding) {
    if (eosTokenId === undefined) {
      errors.push('Missing eos_token_id');
    } else if (Array.isArray(eosTokenId)) {
      if (eosTokenId.length === 0 || eosTokenId.some((id) => typeof id !== 'number')) {
        errors.push('Invalid eos_token_id array');
      }
    } else if (typeof eosTokenId !== 'number') {
      errors.push('Invalid eos_token_id');
    }
  } else if (eosTokenId != null) {
    if (Array.isArray(eosTokenId)) {
      if (eosTokenId.length === 0 || eosTokenId.some((id) => typeof id !== 'number')) {
        errors.push('Invalid eos_token_id array');
      }
    } else if (typeof eosTokenId !== 'number') {
      errors.push('Invalid eos_token_id');
    }
  }

  // Architecture validation (skip for LoRA adapters)
  const isLoRAAdapter = manifest.adapterType === 'lora' || manifest.modelType === 'lora' || !!manifest.loraConfig;

  if (!isLoRAAdapter && !isDiffusion && !isEnergy && !isEmbedding && manifest.architecture && typeof manifest.architecture === 'object') {
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
  } else if (!isLoRAAdapter && !isDiffusion && !isEnergy && !isEmbedding && !manifest.architecture) {
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
      if (tensor.storage !== undefined) {
        validateTensorStorageDescriptor(tensor.storage, `tensor "${name}"`, errors);
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

  // Tensor-config consistency validation
  // This catches bugs like postFeedforwardNorm=false when the weights exist
  if (!isDiffusion && !isEnergy && !isEmbedding) {
    const tensorConfigResult = validateTensorConfigConsistency(manifest);
    for (const err of tensorConfigResult.errors) {
      errors.push(`[${err.code}] ${err.message}${err.suggestion ? ` -> ${err.suggestion}` : ''}`);
    }
    for (const warn of tensorConfigResult.warnings) {
      warnings.push(`[${warn.code}] ${warn.message}${warn.suggestion ? ` -> ${warn.suggestion}` : ''}`);
    }
  }

  if (!isDiffusion && !isEnergy && !isEmbedding && errors.length === 0) {
    try {
      validateManifestInference(manifest);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(message);
    }
  }

  if (!isDiffusion && !isEnergy && !isEmbedding && errors.length === 0) {
    try {
      const executionContract = validateManifestExecutionContract(manifest);
      for (const error of executionContract.errors) {
        errors.push(error);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`[ExecutionContract] ${message}`);
    }
  }

  return { valid: errors.length === 0, errors, warnings };
}
