import { InferencePipeline } from '../text.js';
import { registerPipeline } from '../registry.js';
import { createInitializedPipeline } from '../factory.js';

const DREAM_STRUCTURED_MODEL_TYPES = Object.freeze([
  'dream_structured',
  'dream_intent_posterior_head',
  'dream_d1_to2_bridge',
  'dream_synthesis',
  'dream_energy_compose',
  'dream-intent-posterior-head',
  'dream-d1-to2-bridge',
  'dream-synthesis',
  'dream-energy-compose',
]);

function isObj(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function toHex(bytes) {
  let out = '';
  for (let i = 0; i < bytes.length; i++) {
    out += bytes[i].toString(16).padStart(2, '0');
  }
  return out;
}

async function sha256HexText(text) {
  const payload = String(text ?? '');
  const bytes = new TextEncoder().encode(payload);

  let subtle = globalThis?.crypto?.subtle ?? null;
  if (!subtle) {
    try {
      const nodeCrypto = await import('node:crypto');
      subtle = nodeCrypto?.webcrypto?.subtle ?? null;
    } catch {}
  }
  if (!subtle) {
    throw new Error('DreamStructuredPipeline: SHA-256 requires WebCrypto subtle API.');
  }

  const digest = await subtle.digest('SHA-256', bytes);
  return toHex(new Uint8Array(digest));
}

function parseStructuredJSONObject(rawText) {
  const raw = String(rawText || '');
  const trimmed = raw.trim();
  if (!trimmed) {
    throw new Error('DreamStructuredPipeline: structured decode output is empty.');
  }

  try {
    return JSON.parse(trimmed);
  } catch {}

  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fenced && fenced[1]) {
    try {
      return JSON.parse(String(fenced[1]).trim());
    } catch {}
  }

  const firstBrace = trimmed.indexOf('{');
  const lastBrace = trimmed.lastIndexOf('}');
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    const candidate = trimmed.slice(firstBrace, lastBrace + 1);
    try {
      return JSON.parse(candidate);
    } catch {}
  }

  throw new Error(`DreamStructuredPipeline: invalid JSON decode output (head="${trimmed.slice(0, 96)}").`);
}

function resolveDreamRuntime(manifest, runtimeConfig) {
  const modelCfg = isObj(manifest?.inference?.dream) ? manifest.inference.dream : {};
  const runtimeCfg = isObj(runtimeConfig?.inference?.dream) ? runtimeConfig.inference.dream : {};
  return {
    maxTokens: Number.isFinite(runtimeCfg.maxTokens)
      ? Math.max(1, Math.floor(runtimeCfg.maxTokens))
      : (Number.isFinite(modelCfg.maxTokens) ? Math.max(1, Math.floor(modelCfg.maxTokens)) : 768),
    temperature: Number.isFinite(runtimeCfg.temperature)
      ? Number(runtimeCfg.temperature)
      : (Number.isFinite(modelCfg.temperature) ? Number(modelCfg.temperature) : 0),
    maxOutputChars: Number.isFinite(runtimeCfg.maxOutputChars)
      ? Math.max(4096, Math.floor(runtimeCfg.maxOutputChars))
      : (Number.isFinite(modelCfg.maxOutputChars) ? Math.max(4096, Math.floor(modelCfg.maxOutputChars)) : 262144),
  };
}

export class DreamStructuredPipeline extends InferencePipeline {
  async inferJSON(request = {}) {
    const prompt = String(request?.prompt ?? request?.text ?? '');
    if (!prompt.trim()) {
      throw new Error('DreamStructuredPipeline.inferJSON: prompt is required.');
    }

    const runtime = resolveDreamRuntime(this.manifest, this.runtimeConfig);
    const maxTokens = Number.isFinite(request?.maxTokens)
      ? Math.max(1, Math.floor(request.maxTokens))
      : runtime.maxTokens;
    const temperature = Number.isFinite(request?.temperature)
      ? Number(request.temperature)
      : runtime.temperature;
    const maxOutputChars = Number.isFinite(request?.maxOutputChars)
      ? Math.max(4096, Math.floor(request.maxOutputChars))
      : runtime.maxOutputChars;

    if (typeof this.reset === 'function') {
      this.reset();
    }

    const options = isObj(request?.options) ? { ...request.options } : {};
    options.maxTokens = maxTokens;
    options.temperature = temperature;

    let rawText = '';
    for await (const chunk of this.generate(prompt, options)) {
      rawText += String(chunk || '');
      if (rawText.length > maxOutputChars) {
        throw new Error(`DreamStructuredPipeline.inferJSON: output exceeded ${maxOutputChars} chars.`);
      }
    }

    const output = parseStructuredJSONObject(rawText);
    if (!isObj(output)) {
      throw new Error('DreamStructuredPipeline.inferJSON: output must be a JSON object.');
    }

    const createdAt = String(request?.nowIso || new Date().toISOString());
    const promptHashHex = await sha256HexText(
      JSON.stringify({ prompt, maxTokens, temperature, createdAt })
    );

    return {
      output,
      rawText,
      createdAt,
      modelId: String(this.manifest?.modelId || ''),
      modelHash: this.manifest?.modelHash || null,
      promptHash: { alg: 'sha256', hex: promptHashHex },
    };
  }

  async infer(request = {}) {
    const result = await this.inferJSON(request);
    return result.output;
  }
}

export function isDreamStructuredModelType(modelType) {
  const value = String(modelType || '');
  return DREAM_STRUCTURED_MODEL_TYPES.includes(value);
}

export async function createDreamStructuredPipeline(manifest, contexts = {}) {
  return createInitializedPipeline(DreamStructuredPipeline, manifest, contexts);
}

for (const modelType of DREAM_STRUCTURED_MODEL_TYPES) {
  registerPipeline(modelType, createDreamStructuredPipeline);
}

