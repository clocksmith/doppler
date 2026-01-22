import { ConvertStage } from '../core.js';

const SD3_LAYOUT = {
  id: 'sd3',
  requiredComponents: ['transformer', 'text_encoder', 'text_encoder_2', 'text_encoder_3', 'vae', 'scheduler'],
  weightedComponents: ['transformer', 'text_encoder', 'text_encoder_2', 'text_encoder_3', 'vae'],
  tokenizerSpecs: [
    {
      modelIndexKey: 'tokenizer',
      componentId: 'text_encoder',
      type: 'bpe',
      assets: [
        { suffix: 'tokenizer/vocab.json', targetName: 'tokenizer_vocab.json', kind: 'text', required: true },
        { suffix: 'tokenizer/merges.txt', targetName: 'tokenizer_merges.txt', kind: 'text', required: true },
        { suffix: 'tokenizer/tokenizer_config.json', targetName: 'tokenizer_config.json', kind: 'text', required: false },
        { suffix: 'tokenizer/special_tokens_map.json', targetName: 'tokenizer_special_tokens_map.json', kind: 'text', required: false },
      ],
      config: {
        type: 'bpe',
        vocabFile: 'tokenizer_vocab.json',
        mergesFile: 'tokenizer_merges.txt',
        configFile: 'tokenizer_config.json',
        specialTokensFile: 'tokenizer_special_tokens_map.json',
      },
    },
    {
      modelIndexKey: 'tokenizer_2',
      componentId: 'text_encoder_2',
      type: 'bpe',
      assets: [
        { suffix: 'tokenizer_2/vocab.json', targetName: 'tokenizer_2_vocab.json', kind: 'text', required: true },
        { suffix: 'tokenizer_2/merges.txt', targetName: 'tokenizer_2_merges.txt', kind: 'text', required: true },
        { suffix: 'tokenizer_2/tokenizer_config.json', targetName: 'tokenizer_2_config.json', kind: 'text', required: false },
        { suffix: 'tokenizer_2/special_tokens_map.json', targetName: 'tokenizer_2_special_tokens_map.json', kind: 'text', required: false },
      ],
      config: {
        type: 'bpe',
        vocabFile: 'tokenizer_2_vocab.json',
        mergesFile: 'tokenizer_2_merges.txt',
        configFile: 'tokenizer_2_config.json',
        specialTokensFile: 'tokenizer_2_special_tokens_map.json',
      },
    },
    {
      modelIndexKey: 'tokenizer_3',
      componentId: 'text_encoder_3',
      type: 'sentencepiece',
      assets: [
        { suffix: 'tokenizer_3/tokenizer.json', targetName: 'tokenizer_3_tokenizer.json', kind: 'text', required: true },
        { suffix: 'tokenizer_3/spiece.model', targetName: 'tokenizer_3_spiece.model', kind: 'binary', required: true },
        { suffix: 'tokenizer_3/tokenizer_config.json', targetName: 'tokenizer_3_config.json', kind: 'text', required: false },
        { suffix: 'tokenizer_3/special_tokens_map.json', targetName: 'tokenizer_3_special_tokens_map.json', kind: 'text', required: false },
      ],
      config: {
        type: 'sentencepiece',
        tokenizerFile: 'tokenizer_3_tokenizer.json',
        spieceFile: 'tokenizer_3_spiece.model',
        configFile: 'tokenizer_3_config.json',
        specialTokensFile: 'tokenizer_3_special_tokens_map.json',
      },
    },
  ],
};

const FLUX_LAYOUT = {
  id: 'flux',
  requiredComponents: ['transformer', 'text_encoder', 'vae', 'scheduler'],
  weightedComponents: ['transformer', 'text_encoder', 'vae'],
  tokenizerSpecs: [
    {
      modelIndexKey: 'tokenizer',
      componentId: 'text_encoder',
      type: 'bpe',
      assets: [
        { suffix: 'tokenizer/vocab.json', targetName: 'tokenizer_vocab.json', kind: 'text', required: true },
        { suffix: 'tokenizer/merges.txt', targetName: 'tokenizer_merges.txt', kind: 'text', required: true },
        { suffix: 'tokenizer/tokenizer_config.json', targetName: 'tokenizer_config.json', kind: 'text', required: false },
        { suffix: 'tokenizer/special_tokens_map.json', targetName: 'tokenizer_special_tokens_map.json', kind: 'text', required: false },
        { suffix: 'tokenizer/tokenizer.json', targetName: 'tokenizer_tokenizer.json', kind: 'text', required: false },
        { suffix: 'tokenizer/added_tokens.json', targetName: 'tokenizer_added_tokens.json', kind: 'text', required: false },
        { suffix: 'tokenizer/chat_template.jinja', targetName: 'tokenizer_chat_template.jinja', kind: 'text', required: false },
      ],
      config: {
        type: 'bpe',
        vocabFile: 'tokenizer_vocab.json',
        mergesFile: 'tokenizer_merges.txt',
        configFile: 'tokenizer_config.json',
        specialTokensFile: 'tokenizer_special_tokens_map.json',
      },
    },
  ],
};

const LAYOUTS = [SD3_LAYOUT, FLUX_LAYOUT];

function toAbortError(message = 'Cancelled') {
  if (typeof DOMException === 'function') {
    return new DOMException(message, 'AbortError');
  }
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

function assertNotAborted(signal) {
  if (signal?.aborted) {
    throw toAbortError('Cancelled');
  }
}

function listModelComponents(modelIndex) {
  return Object.keys(modelIndex || {}).filter((key) => !key.startsWith('_'));
}

export function detectDiffusionLayout(modelIndex) {
  const components = new Set(listModelComponents(modelIndex));
  for (const layout of LAYOUTS) {
    if (layout.requiredComponents.every((component) => components.has(component))) {
      return layout;
    }
  }
  const listed = [...components].sort().join(', ') || '(none)';
  const expected = LAYOUTS
    .map((layout) => `${layout.id}: ${layout.requiredComponents.join(', ')}`)
    .join(' | ');
  throw new Error(
    `Unsupported diffusion layout. Found components: ${listed}. Expected one of: ${expected}.`
  );
}

function defaultConfigPath(componentId) {
  if (componentId === 'scheduler') return 'scheduler/scheduler_config.json';
  return `${componentId}/config.json`;
}

function defaultSingleWeightSuffixes(componentId) {
  if (componentId === 'transformer') {
    return [
      'transformer/diffusion_pytorch_model.safetensors',
      'transformer/model.safetensors',
      'transformer/model.fp16.safetensors',
      'sd3.5_medium.safetensors',
    ];
  }
  if (componentId === 'vae') {
    return ['vae/diffusion_pytorch_model.safetensors', 'vae/model.safetensors'];
  }
  return [
    `${componentId}/model.safetensors`,
    `${componentId}/model.fp16.safetensors`,
    `${componentId}/diffusion_pytorch_model.safetensors`,
  ];
}

function defaultIndexSuffixes(componentId) {
  return [
    `${componentId}/model.safetensors.index.json`,
    `${componentId}/model.safetensors.index.fp16.json`,
  ];
}

function addPrefixedTensors(out, componentId, parsed) {
  for (const tensor of parsed.tensors || []) {
    out.push({
      ...tensor,
      name: `${componentId}.${tensor.name}`,
    });
  }
}

export async function parseDiffusionModel(adapter) {
  const {
    readJson,
    readText,
    readBinary,
    findExistingSuffix,
    parseSingleSafetensors,
    parseShardedSafetensors,
    onProgress,
    signal,
  } = adapter;

  assertNotAborted(signal);

  const modelIndex = await readJson('model_index.json', 'model_index.json');
  const layout = detectDiffusionLayout(modelIndex);
  onProgress?.({
    stage: ConvertStage.PARSING,
    message: `Parsing diffusion model_index.json (${layout.id})...`,
  });

  const diffusionConfig = {
    modelIndex,
    layout: layout.id,
    components: {},
    tokenizers: {},
  };
  const auxFiles = [];
  const tensors = [];

  for (const componentId of layout.requiredComponents) {
    const configSuffix = defaultConfigPath(componentId);
    const config = await readJson(configSuffix, `${componentId} config`);
    if (componentId === 'transformer' && config && !config.weight_format) {
      config.weight_format = 'diffusers';
    }
    diffusionConfig.components[componentId] = {
      ...(diffusionConfig.components[componentId] || {}),
      config,
    };
  }

  onProgress?.({
    stage: ConvertStage.PARSING,
    message: `Parsing diffusion weights (${layout.id})...`,
  });

  for (const componentId of layout.weightedComponents) {
    const singleSuffix = findExistingSuffix(defaultSingleWeightSuffixes(componentId));
    const indexSuffix = findExistingSuffix(defaultIndexSuffixes(componentId));
    if (!singleSuffix && !indexSuffix) {
      throw new Error(`Missing ${componentId} safetensors file(s)`);
    }
    if (singleSuffix && indexSuffix) {
      throw new Error(
        `Ambiguous ${componentId} weights: both single and sharded files present. Keep one format.`
      );
    }
    if (indexSuffix) {
      const indexJson = await readJson(indexSuffix, `${componentId} index`);
      const parsed = await parseShardedSafetensors(indexSuffix, indexJson, componentId);
      addPrefixedTensors(tensors, componentId, parsed);
    } else {
      const parsed = await parseSingleSafetensors(singleSuffix, componentId);
      addPrefixedTensors(tensors, componentId, parsed);
    }
  }

  for (const tokenizerSpec of layout.tokenizerSpecs) {
    const enabled = Boolean(modelIndex?.[tokenizerSpec.modelIndexKey]);
    if (!enabled) continue;
    for (const asset of tokenizerSpec.assets) {
      const existing = findExistingSuffix([asset.suffix]);
      if (!existing) {
        if (asset.required) {
          throw new Error(
            `Missing ${tokenizerSpec.componentId} tokenizer asset (${asset.suffix}) for diffusion conversion.`
          );
        }
        continue;
      }
      if (asset.kind === 'binary') {
        auxFiles.push({ name: asset.targetName, data: await readBinary(existing, asset.targetName) });
      } else {
        auxFiles.push({ name: asset.targetName, data: await readText(existing, asset.targetName) });
      }
    }
    diffusionConfig.tokenizers[tokenizerSpec.componentId] = { ...tokenizerSpec.config };
  }

  assertNotAborted(signal);

  return {
    tensors,
    config: { diffusion: diffusionConfig },
    auxFiles,
    architecture: 'diffusion',
    layout: layout.id,
  };
}
