import fs from 'node:fs';
import path from 'node:path';

const SYNTHETIC_PROMPT_FRAGMENTS = Object.freeze([
  'the',
  'system',
  'should',
  'allow',
  'review',
  'when',
  'policy',
  'is',
  'clear',
  'and',
  'block',
  'unsafe',
  'changes',
  'only',
  'if',
  'evidence',
  'supports',
  'it',
  'user',
  'request',
  'must',
  'stay',
  'safe',
  'true',
  'false',
  'yes',
  'no',
  'a',
]);

const MAX_SYNTHETIC_PROMPT_STEPS_PER_TOKEN = 8;

function normalizePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function normalizeEncodedLength(encoded, label) {
  if (Array.isArray(encoded) || ArrayBuffer.isView(encoded)) {
    return encoded.length;
  }
  const inputIds = encoded?.input_ids;
  if (Array.isArray(inputIds) || ArrayBuffer.isView(inputIds)) {
    return inputIds.length;
  }
  if (Number.isInteger(inputIds?.size)) {
    return inputIds.size;
  }
  throw new Error(`${label} must resolve to a token id array or tensor-like input_ids payload.`);
}

function normalizeModelLocator(modelId, localModelPath = null) {
  if (typeof localModelPath === 'string' && localModelPath.trim() !== '') {
    const localModelRoot = path.resolve(localModelPath);
    const normalizedModelId = typeof modelId === 'string' && modelId.trim() !== ''
      ? modelId.trim()
      : null;
    const localModelDir = normalizedModelId
      ? path.join(localModelRoot, normalizedModelId)
      : null;
    if (localModelDir && fs.existsSync(localModelDir)) {
      return {
        locator: localModelDir,
        source: 'local-model-root',
      };
    }
    return {
      locator: localModelRoot,
      source: 'local-model-path',
    };
  }
  if (typeof modelId !== 'string' || modelId.trim() === '') {
    throw new Error('Tokenizer-backed synthetic prompt resolution requires a non-empty model id or local model path.');
  }
  return {
    locator: modelId.trim(),
    source: 'model-id',
  };
}

async function importAutoTokenizer() {
  const module = await import('@huggingface/transformers');
  if (typeof module?.AutoTokenizer?.from_pretrained !== 'function') {
    throw new Error('@huggingface/transformers AutoTokenizer.from_pretrained is unavailable.');
  }
  return module.AutoTokenizer;
}

function readLocalChatTemplate(tokenizerLocator, tokenizerResolutionSource) {
  if (
    typeof tokenizerLocator !== 'string'
    || tokenizerLocator.trim() === ''
    || typeof tokenizerResolutionSource !== 'string'
    || !tokenizerResolutionSource.startsWith('local-model')
  ) {
    return null;
  }
  const chatTemplatePath = path.join(tokenizerLocator, 'chat_template.jinja');
  if (!fs.existsSync(chatTemplatePath)) {
    return null;
  }
  const chatTemplate = fs.readFileSync(chatTemplatePath, 'utf8');
  const normalized = String(chatTemplate ?? '').trim();
  return normalized === '' ? null : normalized;
}

export async function createTokenizerPromptCounter({
  modelId,
  localModelPath = null,
  useChatTemplate = false,
}) {
  const primaryLocator = normalizeModelLocator(modelId, localModelPath);
  const AutoTokenizer = await importAutoTokenizer();
  let tokenizer = null;
  let tokenizerLocator = primaryLocator.locator;
  let tokenizerResolutionSource = primaryLocator.source;
  try {
    tokenizer = await AutoTokenizer.from_pretrained(primaryLocator.locator);
  } catch (primaryError) {
    const fallbackModelId = typeof modelId === 'string' && modelId.trim() !== ''
      ? modelId.trim()
      : null;
    const canFallbackToModelId = fallbackModelId != null
      && fallbackModelId !== primaryLocator.locator
      && (primaryLocator.source === 'local-model-root' || primaryLocator.source === 'local-model-path');
    if (!canFallbackToModelId) {
      const message = primaryError instanceof Error ? primaryError.message : String(primaryError);
      throw new Error(
        `Failed to load tokenizer for synthetic prompt resolution from "${primaryLocator.locator}": ${message}`
      );
    }
    try {
      tokenizer = await AutoTokenizer.from_pretrained(fallbackModelId);
      tokenizerLocator = fallbackModelId;
      tokenizerResolutionSource = `${primaryLocator.source}-fallback-model-id`;
    } catch (fallbackError) {
      const primaryMessage = primaryError instanceof Error ? primaryError.message : String(primaryError);
      const fallbackMessage = fallbackError instanceof Error ? fallbackError.message : String(fallbackError);
      throw new Error(
        `Failed to load tokenizer for synthetic prompt resolution from "${primaryLocator.locator}" `
        + `(${primaryMessage}) and fallback model id "${fallbackModelId}" (${fallbackMessage}).`
      );
    }
  }

  const localChatTemplate = readLocalChatTemplate(tokenizerLocator, tokenizerResolutionSource);
  const countPromptTokens = async (promptText) => {
    const prompt = String(promptText ?? '');
    if (useChatTemplate === true) {
      if (typeof tokenizer?.apply_chat_template !== 'function') {
        throw new Error(
          `Tokenizer "${tokenizerLocator}" does not expose apply_chat_template, but chat-template prompt targeting was requested.`
        );
      }
      const chatTemplateOverride = typeof tokenizer?.chat_template === 'string' && tokenizer.chat_template.trim() !== ''
        ? tokenizer.chat_template
        : localChatTemplate;
      const rendered = await tokenizer.apply_chat_template(
        [{ role: 'user', content: prompt }],
        {
          tokenize: true,
          add_generation_prompt: true,
          return_tensor: false,
          ...(chatTemplateOverride ? { chat_template: chatTemplateOverride } : {}),
        }
      );
      return normalizeEncodedLength(rendered, `apply_chat_template(${tokenizerLocator})`);
    }
    const encoded = await tokenizer.encode(prompt);
    return normalizeEncodedLength(encoded, `tokenizer.encode(${tokenizerLocator})`);
  };

  return {
    countPromptTokens,
    tokenizerLocator,
    tokenizerResolutionSource,
  };
}

export async function buildTokenAccurateSyntheticPrompt({
  prefillTokens,
  countPromptTokens,
  fragments = SYNTHETIC_PROMPT_FRAGMENTS,
}) {
  const targetTokens = normalizePositiveInteger(prefillTokens, 'prefillTokens');
  if (typeof countPromptTokens !== 'function') {
    throw new Error('countPromptTokens must be a function.');
  }
  if (!Array.isArray(fragments) || fragments.length < 1) {
    throw new Error('fragments must be a non-empty array.');
  }

  let prompt = '';
  let currentTokens = await countPromptTokens(prompt);
  if (!Number.isInteger(currentTokens) || currentTokens < 0) {
    throw new Error(`countPromptTokens("") must return a non-negative integer; got ${String(currentTokens)}.`);
  }
  if (currentTokens > targetTokens) {
    throw new Error(
      `Synthetic prompt target ${targetTokens} is smaller than the tokenizer baseline ${currentTokens}; `
      + 'increase the target or disable chat-template prompt budgeting.'
    );
  }

  let cursor = 0;
  const maxSteps = Math.max(targetTokens, 1) * MAX_SYNTHETIC_PROMPT_STEPS_PER_TOKEN;
  for (let step = 0; step < maxSteps && currentTokens < targetTokens; step += 1) {
    const remaining = targetTokens - currentTokens;
    let chosen = null;

    for (let offset = 0; offset < fragments.length; offset += 1) {
      const index = (cursor + offset) % fragments.length;
      const baseFragment = fragments[index];
      if (typeof baseFragment !== 'string' || baseFragment.trim() === '') {
        throw new Error(`fragments[${index}] must be a non-empty string.`);
      }
      const fragment = prompt.length === 0 ? baseFragment : ` ${baseFragment}`;
      const nextPrompt = `${prompt}${fragment}`;
      const nextTokens = await countPromptTokens(nextPrompt);
      if (!Number.isInteger(nextTokens) || nextTokens < currentTokens) {
        throw new Error(
          `countPromptTokens("${baseFragment}") returned invalid token count ${String(nextTokens)}.`
        );
      }
      const delta = nextTokens - currentTokens;
      if (delta < 1 || delta > remaining) {
        continue;
      }
      if (chosen == null || delta > chosen.delta || (delta === chosen.delta && offset < chosen.offset)) {
        chosen = {
          prompt: nextPrompt,
          tokens: nextTokens,
          delta,
          offset,
          nextCursor: (index + 1) % fragments.length,
        };
      }
      if (delta === remaining) {
        break;
      }
    }

    if (chosen == null) {
      break;
    }
    prompt = chosen.prompt;
    currentTokens = chosen.tokens;
    cursor = chosen.nextCursor;
  }

  if (currentTokens !== targetTokens) {
    throw new Error(
      `Could not synthesize an exact ${targetTokens}-token prompt; resolved ${currentTokens} tokens instead.`
    );
  }

  return {
    prompt,
    prefillTokens: currentTokens,
    promptSource: 'tokenizer-accurate-synthetic',
  };
}

export async function resolveSyntheticPromptForModel({
  prefillTokens,
  modelId,
  localModelPath = null,
  useChatTemplate = false,
}) {
  const counter = await createTokenizerPromptCounter({
    modelId,
    localModelPath,
    useChatTemplate,
  });
  const prompt = await buildTokenAccurateSyntheticPrompt({
    prefillTokens,
    countPromptTokens: counter.countPromptTokens,
  });
  return {
    ...prompt,
    tokenizerLocator: counter.tokenizerLocator,
    tokenizerResolutionSource: counter.tokenizerResolutionSource,
  };
}
