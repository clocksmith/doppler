function asNonEmptyString(value, label) {
  const normalized = typeof value === 'string' ? value.trim() : '';
  if (!normalized) {
    throw new Error(`transformer parser: ${label} must be a non-empty string.`);
  }
  return normalized;
}

function isSentenceTransformerModuleType(type, expectedSuffix) {
  return typeof type === 'string' && type.trim().endsWith(expectedSuffix);
}

function resolvePoolingMode(config) {
  const supported = [
    ['mean', config?.pooling_mode_mean_tokens === true],
    ['last', config?.pooling_mode_lasttoken === true],
  ].filter(([, enabled]) => enabled);
  if (supported.length !== 1) {
    throw new Error(
      'transformer parser: sentence-transformers pooling must enable exactly one supported mode (mean or lasttoken).'
    );
  }
  if (
    config?.pooling_mode_cls_token === true
    || config?.pooling_mode_max_tokens === true
    || config?.pooling_mode_mean_sqrt_len_tokens === true
    || config?.pooling_mode_weightedmean_tokens === true
  ) {
    throw new Error(
      'transformer parser: unsupported sentence-transformers pooling mode; only mean or lasttoken pooling is supported.'
    );
  }
  return supported[0][0];
}

function resolveDenseActivation(config) {
  const raw = typeof config?.activation_function === 'string'
    ? config.activation_function.trim()
    : '';
  if (raw === 'torch.nn.modules.linear.Identity') {
    return 'identity';
  }
  throw new Error(
    `transformer parser: unsupported sentence-transformers Dense activation "${raw || 'unknown'}".`
  );
}

function renameDenseTensor(tensor, tensorName) {
  return {
    ...tensor,
    name: tensorName,
    role: 'lm_head',
    group: 'embedding_postprocessor',
  };
}

async function parseSentenceTransformersModules(adapter) {
  const {
    readJson,
    loadSingleSafetensors,
  } = adapter;
  const modules = await readJson('modules.json', 'modules.json');
  if (!Array.isArray(modules) || modules.length === 0) {
    throw new Error('transformer parser: modules.json must contain a non-empty array.');
  }

  let poolingMode = null;
  let includePrompt = true;
  let normalize = null;
  const projections = [];
  const extraTensors = [];

  for (const moduleEntry of modules) {
    const type = asNonEmptyString(moduleEntry?.type, 'modules.json type');
    const modulePath = typeof moduleEntry?.path === 'string'
      ? moduleEntry.path.trim()
      : '';

    if (isSentenceTransformerModuleType(type, '.Transformer')) {
      continue;
    }

    if (isSentenceTransformerModuleType(type, '.Pooling')) {
      const config = await readJson(`${modulePath}/config.json`, `${modulePath}/config.json`);
      poolingMode = resolvePoolingMode(config);
      includePrompt = config?.include_prompt !== false;
      continue;
    }

    if (isSentenceTransformerModuleType(type, '.Dense')) {
      const projectionIndex = projections.length;
      const config = await readJson(`${modulePath}/config.json`, `${modulePath}/config.json`);
      const inputSize = Number(config?.in_features);
      const outputSize = Number(config?.out_features);
      if (!Number.isFinite(inputSize) || inputSize <= 0 || !Number.isFinite(outputSize) || outputSize <= 0) {
        throw new Error(`transformer parser: ${modulePath}/config.json must declare positive in_features/out_features.`);
      }
      const hasBias = config?.bias === true;
      const activation = resolveDenseActivation(config);
      const denseTensors = await loadSingleSafetensors(`${modulePath}/model.safetensors`);
      const weightTensorName = `embedding_postprocessor.projections.${projectionIndex}.weight`;
      const biasTensorName = hasBias
        ? `embedding_postprocessor.projections.${projectionIndex}.bias`
        : null;
      let foundWeight = false;
      let foundBias = false;
      for (const tensor of denseTensors) {
        const sourceName = asNonEmptyString(tensor?.name, `${modulePath} tensor name`);
        if (sourceName === 'linear.weight') {
          extraTensors.push(renameDenseTensor(tensor, weightTensorName));
          foundWeight = true;
          continue;
        }
        if (sourceName === 'linear.bias' && hasBias && biasTensorName) {
          extraTensors.push(renameDenseTensor(tensor, biasTensorName));
          foundBias = true;
          continue;
        }
        throw new Error(`transformer parser: unsupported Dense tensor "${sourceName}" in ${modulePath}/model.safetensors.`);
      }
      if (!foundWeight) {
        throw new Error(`transformer parser: ${modulePath}/model.safetensors is missing linear.weight.`);
      }
      if (hasBias && !foundBias) {
        throw new Error(`transformer parser: ${modulePath}/model.safetensors is missing linear.bias.`);
      }
      projections.push({
        weightTensor: weightTensorName,
        biasTensor: biasTensorName,
        inputSize: Math.trunc(inputSize),
        outputSize: Math.trunc(outputSize),
        activation,
      });
      continue;
    }

    if (isSentenceTransformerModuleType(type, '.Normalize')) {
      normalize = 'l2';
      continue;
    }

    throw new Error(`transformer parser: unsupported sentence-transformers module type "${type}".`);
  }

  if (poolingMode == null && projections.length === 0 && normalize == null) {
    return {
      extraTensors: [],
      embeddingPostprocessor: null,
    };
  }
  if (poolingMode == null) {
    throw new Error('transformer parser: sentence-transformers modules require a supported Pooling module.');
  }

  return {
    extraTensors,
    embeddingPostprocessor: {
      poolingMode,
      includePrompt,
      projections,
      normalize,
    },
  };
}

export async function parseTransformerModel(adapter) {
  const {
    readJson,
    fileExists,
    loadSingleSafetensors,
    loadShardedSafetensors,
  } = adapter;

  const config = await readJson('config.json', 'config.json');
  const generationConfig = await fileExists('generation_config.json')
    ? await readJson('generation_config.json', 'generation_config.json')
    : null;
  const architectureHint = config.architectures?.[0] ?? config.model_type ?? '';
  const sentenceTransformersModules = await fileExists('modules.json')
    ? await parseSentenceTransformersModules(adapter)
    : { extraTensors: [], embeddingPostprocessor: null };

  let tensors = null;
  if (await fileExists('model.safetensors.index.json')) {
    const indexJson = await readJson('model.safetensors.index.json', 'model.safetensors.index.json');
    tensors = await loadShardedSafetensors(indexJson);
  } else {
    tensors = await loadSingleSafetensors('model.safetensors');
  }

  return {
    config,
    generationConfig,
    tensors: [...tensors, ...sentenceTransformersModules.extraTensors],
    architectureHint,
    embeddingPostprocessor: sentenceTransformersModules.embeddingPostprocessor,
  };
}
