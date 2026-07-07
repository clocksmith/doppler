#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

const TEXT_FILES = Object.freeze([
  'src/config/schema/kernel-path.schema.d.ts',
  'src/config/schema/inference.schema.d.ts',
  'src/config/schema/index.d.ts',
  'src/config/kernel-path-loader.d.ts',
  'src/config/README.md',
  'docs/config.md',
  'docs/cli.md',
  'docs/style/general-style-guide.md',
  'docs/style/benchmark-style-guide.md',
  'docs/style/config-style-guide.md',
  'docs/developer-guides/config-source-of-truth.md',
  'docs/model-promotion-playbook.md',
  'docs/developer-guides/06-kernel-path-config.md',
  'docs/developer-guides/11-wgsl-kernel.md',
  'docs/developer-guides/05-promote-model-artifact.md',
  'docs/developer-guides/13-attention-variant.md',
  'src/gpu/kernels/README.md',
  'docs/registry-workflow.md',
  'models/README.md',
]);

const RULE_JSON_FILES = Object.freeze([
  'src/rules/inference/config.rules.json',
  'src/rules/inference/execution.rules.json',
  'src/rules/inference/capability-transforms.rules.json',
  'src/rules/inference/dtype.rules.json',
]);

const MODEL_IDENTITY_RULE_KEYS = Object.freeze(new Set([
  'modelId',
  'manifestModelType',
]));

const FORBIDDEN_MODEL_IDENTITY_OPERATORS = Object.freeze([
  'contains',
  'startsWith',
  'endsWith',
]);

const TEXT_RULES = Object.freeze([
  {
    label: 'removed kernel-path asset directory',
    pattern: /src\/config\/kernel-paths/g,
  },
  {
    label: 'registered kernel-path ID wording',
    pattern: /registered kernel-path ID/gi,
  },
  {
    label: 'KernelPathRef string union',
    pattern: /KernelPathRef\s*=\s*string\s*\|/g,
  },
  {
    label: 'BuiltinKernelPathId export',
    pattern: /BuiltinKernelPathId/g,
  },
  {
    label: 'models/local source-of-truth wording',
    pattern: /models\/local\/[^\n]*source of truth/gi,
  },
  {
    label: 'stale two-file registry architecture wording',
    pattern: /Two files, one direction/gi,
  },
  {
    label: 'entry-point-over-override blanket kernel guidance',
    pattern: /DOPPLER uses entry points[\s\S]{0,120}override constants/gi,
  },
  {
    label: 'removed external-index script workflow',
    pattern: /npm run external:index|tools\/sync-external-rdrr-index\.js/g,
  },
]);

const REQUIRED_TEXT_SNIPPETS = Object.freeze({
  'docs/developer-guides/config-source-of-truth.md': [
    'src/config/kernels/registry.json',
    'models/catalog.json',
    'src/rules/inference/capability-transforms.rules.json',
    'npm run capability-policy:check',
    'npm run kernels:registry:check',
    'npm run support:inventory:check',
    'npm run ci:catalog:check',
    'npm run registry:hf:check',
    'npm run artifact:contract:check',
  ],
  'src/gpu/kernels/README.md': [
    'docs/developer-guides/config-source-of-truth.md',
    'src/config/kernels/registry.json',
    'npm run kernels:registry:check',
  ],
  'docs/registry-workflow.md': [
    'models/catalog.json',
    'npm run support:inventory:check',
    'npm run ci:catalog:check',
    'developer-guides/config-source-of-truth.md',
  ],
  'models/README.md': [
    'models/catalog.json',
    'docs/model-support-inventory.md',
    'npm run ci:catalog:check',
    'docs/developer-guides/config-source-of-truth.md',
  ],
});

const CODE_RULES = Object.freeze([
  {
    file: 'src/gpu/kernels/matmul.js',
    label: 'Matmul output dtype derived from input dtype',
    pattern: /options\.outputDtype\s*\|\|\s*A\.dtype|requestedOutputDtype\s*=\s*options\.outputDtype\s*\|\|/g,
  },
  {
    file: 'src/gpu/kernels/matmul.js',
    label: 'Matmul missing weight dtype warning fallback',
    pattern: /Assuming f32/g,
  },
  {
    file: 'src/gpu/kernels/linear-attention-core.js',
    label: 'Linear attention output dtype runtime fallback',
    pattern: /outputDtype\s*===\s*undefined\s*\?\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/weights.js',
    label: 'Text weight buffer dtype runtime fallback',
    pattern: /weight\.dtype\s*\?\?\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/embed.js',
    label: 'Embedding range-backed dtype runtime fallback',
    pattern: /sourceDtype\s*\?\?\s*embedBuffer\.dtype\s*\?\?\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/per-layer-inputs.js',
    label: 'Per-layer input projection norm dtype runtime fallback',
    pattern: /getPleProjectionNormDtype[\s\S]{0,120}\?\?\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/per-layer-inputs.js',
    label: 'Per-layer input source dtype runtime fallback',
    pattern: /embedTokensPerLayer[\s\S]{0,160}\?\?\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/moe-shape-validator.js',
    label: 'MoE validator router dtype runtime fallback',
    pattern: /routerDtype:\s*context\?\.routerDtype\s*\?\?\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/moe-shape-validator.js',
    label: 'MoE validator output dtype runtime fallback',
    pattern: /outputDtype:\s*context\?\.outputDtype\s*\?\?\s*context\?\.weightsDtype/g,
  },
  {
    file: 'src/inference/pipelines/text/logits/precision-policy.js',
    label: 'Logits local kernel remap table',
    pattern: /STABLE_F32_LOGITS_KERNEL_MAP|new Map\(\[/g,
  },
  {
    file: 'src/inference/kv-cache/quantized.js',
    label: 'QuantizedKVCache kvDtype runtime fallback',
    pattern: /config\.kvDtype\s*\?\?\s*['"]f16['"]/g,
  },
  {
    file: 'src/inference/kv-cache/quantized.js',
    label: 'QuantizedKVCache bitWidth runtime fallback',
    pattern: /config\.bitWidth\s*\?\?\s*4/g,
  },
  {
    file: 'src/inference/kv-cache/tiered.js',
    label: 'TieredKVCache coldDtype runtime fallback',
    pattern: /tiering\.coldDtype\s*\?\?\s*this\.kvDtype/g,
  },
  {
    file: 'src/inference/kv-cache/tiered.js',
    label: 'TieredKVCache compression object runtime fallback',
    pattern: /tiering\.compression\s*\?\?/g,
  },
  {
    file: 'src/inference/kv-cache/tiered.js',
    label: 'TieredKVCache gating object runtime fallback',
    pattern: /tiering\.gating\s*\?\?/g,
  },
  {
    file: 'src/inference/kv-cache/tiered.js',
    label: 'TieredKVCache compression mode runtime fallback',
    pattern: /compression\?\.mode\s*\?\?\s*['"]none['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/init.js',
    label: 'KV tiering mode runtime fallback',
    pattern: /tiering\?\.mode\s*\?\?\s*['"]off['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/init.js',
    label: 'KV compression mode runtime fallback',
    pattern: /compression\?\.mode\s*\?\?/g,
  },
  {
    file: 'src/inference/pipelines/text/init.js',
    label: 'KV quantization mode runtime fallback',
    pattern: /quantization\?\.mode\s*\?\?\s*['"]none['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/init.js',
    label: 'Quantized KV bitWidth runtime fallback',
    pattern: /quantCfg\.bitWidth\s*\?\?\s*4/g,
  },
  {
    file: 'src/gpu/kernels/turboquant-codebook.js',
    label: 'TurboQuant shared buffer bitWidth runtime fallback',
    pattern: /options\.bitWidth\s*\?\?\s*4/g,
  },
  {
    file: 'src/inference/pipelines/text.js',
    label: 'Multimodal generation maxTokens runtime fallback',
    pattern: /maxTokens\s*\?\?\s*512/g,
  },
  {
    file: 'src/inference/pipelines/text/execution-plan.js',
    label: 'Execution plan maxTokens runtime fallback',
    pattern: /DEFAULT_MAX_TOKENS|generationConfig\.maxTokens\s*\?\?/g,
  },
  {
    file: 'src/gpu/profiler.js',
    label: 'Profiler maxHistoryLabels runtime fallback',
    pattern: /maxHistoryLabels\s*\?\?\s*1024/g,
  },
  {
    file: 'src/gpu/kernels/matmul-selection.js',
    label: 'Matmul selector output dtype runtime fallback',
    pattern: /outputDtype\s*=\s*['"]f32['"]/g,
  },
  {
    file: 'src/gpu/kernels/dequant.js',
    label: 'Dequant output dtype runtime fallback',
    pattern: /outputDtype\s*=\s*['"]f32['"]|outputDtype\s*=\s*['"]f16['"]/g,
  },
  {
    file: 'src/gpu/kernels/softmax.js',
    label: 'SoftmaxTopK dtype runtime fallback',
    pattern: /inputDtype\s*=\s*['"]f32['"]|weightsDtype\s*=\s*['"]f32['"]/g,
  },
  {
    file: 'src/gpu/kernels/moe.js',
    label: 'MoE scatter weights dtype runtime fallback',
    pattern: /weightsDtype\s*=\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/config.js',
    label: 'Audio hidden activation runtime fallback',
    pattern: /hidden_act\s*\?\?\s*['"]silu['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/config.js',
    label: 'Runtime model ID family detection',
    pattern: /modelId[\s\S]{0,160}\.includes\s*\(\s*['"]lfm2['"]\s*\)/g,
  },
  {
    file: 'src/inference/pipelines/text/ffn/moe.js',
    label: 'MoE modelType inference from expertFormat',
    pattern: /modelType:\s*config\.modelType\s*\?\?|expertFormat\s*===\s*['"]gpt-oss['"][\s\S]{0,120}expertFormat\s*===\s*['"]mixtral['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/moe-gpu.js',
    label: 'MoE GPU expertFormat runtime branch',
    pattern: /expertFormat\s*===\s*['"](gpt-oss|gemma4|mixtral)['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/moe-gpu.js',
    label: 'MoE GPU modelType runtime branch',
    pattern: /modelType\s*===\s*['"](gpt-oss|gemma4|diffusion_gemma|mixtral)['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/moe-shape-validator.js',
    label: 'MoE validator modelType inference from expertFormat',
    pattern: /options\.modelType\s*\?\?|expertFormat\s*===\s*['"]gpt-oss['"][\s\S]{0,120}expertFormat\s*===\s*['"]gemma4['"]/g,
  },
  {
    file: 'src/inference/browser-harness-text-helpers.js',
    label: 'Browser harness generation defaults',
    pattern: /DEFAULT_HARNESS_PROMPT|DEFAULT_HARNESS_MAX_TOKENS|DEFAULT_SAMPLING_DEFAULTS|Running with default inference parameters|warnIfUsingDefaults|buildDefaultGenerationPrompt|shouldPreferModelDefaultPrompt/g,
  },
  {
    file: 'src/inference/pipelines/text/config.js',
    label: 'FFN branchMode runtime fallback',
    pattern: /branchMode\s*\?\?\s*['"]auto['"]/g,
  },
  {
    file: 'src/rules/execution-rules-contract-check.js',
    label: 'Execution rule semantic model ID substring detection',
    pattern: /modelId\.includes\s*\(/g,
  },
  {
    file: 'src/inference/pipelines/diffusion/text-encoder-gpu.js',
    label: 'CLIP hidden activation runtime fallback',
    pattern: /DEFAULT_CLIP_HIDDEN_ACT|hidden_act\s*\?\?\s*['"]gelu['"]/g,
  },
  {
    file: 'src/config/execution-contract-check.js',
    label: 'Execution contract tiering mode runtime fallback',
    pattern: /tiering\?\.mode\s*\?\?\s*['"]off['"]/g,
  },
  {
    file: 'src/config/execution-contract-check.js',
    label: 'Execution contract gating mode runtime fallback',
    pattern: /gating\?\.mode\s*\?\?\s*['"]auto['"]/g,
  },
  {
    file: 'src/config/execution-contract-check.js',
    label: 'Execution contract compression mode runtime fallback',
    pattern: /compression\?\.mode\s*\?\?/g,
  },
  {
    file: 'src/config/execution-contract-check.js',
    label: 'Execution contract quantization mode runtime fallback',
    pattern: /quantization\?\.mode\s*\?\?\s*['"]none['"]/g,
  },
  {
    file: 'src/inference/moe-router.js',
    label: 'MoE router input dtype runtime fallback',
    pattern: /inputDtype\s*=\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/moe-router.js',
    label: 'MoE router output dtype runtime fallback',
    pattern: /outputDtype\s*=\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/moe-router.js',
    label: 'MoE router last logits dtype runtime fallback',
    pattern: /lastLogitsDtype\s*=\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/generator-runtime.js',
    label: 'Generator runtime local dtype default',
    pattern: /DEFAULT_DTYPE|fallback:\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/generator-runtime.js',
    label: 'Generator self-speculation runtime fallback',
    pattern: /sessionSpeculation\?\.mode\s*\?\?\s*['"]none['"]|sessionSpeculation\?\.tokens\s*\?\?\s*1|sessionSpeculation\?\.verify\s*\?\?\s*['"]greedy['"]|sessionSpeculation\?\.rollbackOnReject\s*\?\?\s*true/g,
  },
  {
    file: 'src/inference/pipelines/text/generator.js',
    label: 'Self-speculation token runtime fallback',
    pattern: /speculation\?\.tokens\s*\?\?\s*1/g,
  },
  {
    file: 'src/inference/pipelines/text/logits/gpu.js',
    label: 'LM head CPU source dtype runtime fallback',
    pattern: /sourceDtype\s*=\s*['"]f32['"]|sourceDtype\s*\?\?\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/logits/index.js',
    label: 'Stable F32 logits inline policy',
    pattern: /finalLogitSoftcapping\)\s*&&\s*config\.finalLogitSoftcapping\s*>\s*0|hiddenSize\s*<=\s*768/g,
  },
  {
    file: 'src/inference/pipelines/text/logits/gpu.js',
    label: 'Stable F32 logits inline policy',
    pattern: /finalLogitSoftcapping\)\s*&&\s*config\.finalLogitSoftcapping\s*>\s*0|hiddenSize\s*<=\s*768/g,
  },
  {
    file: 'src/inference/pipelines/text/ffn/dense.js',
    label: 'FFN fused gate dtype runtime fallback',
    pattern: /gateDtype\s*=\s*fusedGateUpWeights\.gateDtype\s*\?\?\s*['"]f32['"]|gateDtype\s*=\s*fusedGateUpWeights\.gateDtype\s*\?\?\s*\(hasGate\s*\?\s*['"]f32['"]/g,
  },
  {
    file: 'src/inference/pipelines/text/ffn/dense.js',
    label: 'FFN fused up dtype runtime fallback',
    pattern: /upDtype\s*=\s*fusedGateUpWeights\.upDtype\s*\?\?\s*['"]f32['"]|upDtype\s*=\s*fusedGateUpWeights\.upDtype\s*\?\?\s*\(hasUp\s*\?\s*['"]f32['"]/g,
  },
]);

const issues = [];

function toRepoPath(filePath) {
  return path.relative(ROOT, filePath).replace(/\\/g, '/');
}

function recordIssue(filePath, message) {
  issues.push(`${toRepoPath(filePath)}: ${message}`);
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function collectJsonFiles(dirPath) {
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      files.push(...await collectJsonFiles(fullPath));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.json')) {
      files.push(fullPath);
    }
  }
  return files;
}

function scanRuntimeJsonValue(filePath, value, keyPath = []) {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => scanRuntimeJsonValue(filePath, entry, keyPath.concat(String(index))));
    return;
  }
  if (!value || typeof value !== 'object') return;

  for (const [key, child] of Object.entries(value)) {
    const nextPath = keyPath.concat(key);
    if (key === 'kernelPlan') {
      recordIssue(filePath, `${nextPath.join('.')} uses removed kernelPlan`);
    }
    if (key === 'kernelPath' && typeof child === 'string') {
      recordIssue(filePath, `${nextPath.join('.')} uses removed string kernel-path ID "${child}"`);
    }
    scanRuntimeJsonValue(filePath, child, nextPath);
  }
}

function scanRuleJsonValue(filePath, value, keyPath = []) {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => scanRuleJsonValue(filePath, entry, keyPath.concat(String(index))));
    return;
  }
  if (!value || typeof value !== 'object') return;

  for (const [key, child] of Object.entries(value)) {
    const nextPath = keyPath.concat(key);
    if (MODEL_IDENTITY_RULE_KEYS.has(key) && child && typeof child === 'object' && !Array.isArray(child)) {
      for (const operator of FORBIDDEN_MODEL_IDENTITY_OPERATORS) {
        if (Object.prototype.hasOwnProperty.call(child, operator)) {
          recordIssue(
            filePath,
            `${nextPath.join('.')} uses ${operator}; model identity rules must use exact values or explicit lists`
          );
        }
      }
    }
    scanRuleJsonValue(filePath, child, nextPath);
  }
}

function checkDtypeRuleFallbacks(filePath, parsed) {
  const fallbackRequiredRules = [
    'f16OrF32FromDtype',
    'bytesPerElement',
    'dtypeFromSize',
    'bytesFromDtype',
    'matmulDtype',
  ];
  for (const ruleName of fallbackRequiredRules) {
    const entries = parsed?.[ruleName];
    if (!Array.isArray(entries)) {
      recordIssue(filePath, `dtype rule ${ruleName} is missing`);
      continue;
    }
    const catchall = entries.find((entry) => {
      const match = entry?.match;
      return match && typeof match === 'object' && !Array.isArray(match) && Object.keys(match).length === 0;
    });
    if (!catchall) {
      recordIssue(filePath, `dtype rule ${ruleName} must have an explicit catchall`);
      continue;
    }
    if (catchall.value?.context !== 'fallback') {
      recordIssue(filePath, `dtype rule ${ruleName} catchall must read { "context": "fallback" }`);
    }
  }
}

async function checkTextFiles() {
  for (const relPath of TEXT_FILES) {
    const filePath = path.join(ROOT, relPath);
    if (!await pathExists(filePath)) {
      recordIssue(filePath, 'expected contract file is missing');
      continue;
    }
    const text = await fs.readFile(filePath, 'utf8');
    for (const rule of TEXT_RULES) {
      const matches = text.match(rule.pattern);
      if (matches?.length) {
        recordIssue(filePath, `${rule.label} appears ${matches.length} time(s)`);
      }
    }
    const requiredSnippets = REQUIRED_TEXT_SNIPPETS[relPath] || [];
    for (const snippet of requiredSnippets) {
      if (!text.includes(snippet)) {
        recordIssue(filePath, `missing required single-source reference: ${snippet}`);
      }
    }
  }
}

async function checkRuleJsonFiles() {
  for (const relPath of RULE_JSON_FILES) {
    const filePath = path.join(ROOT, relPath);
    if (!await pathExists(filePath)) {
      recordIssue(filePath, 'expected rule file is missing');
      continue;
    }
    const text = await fs.readFile(filePath, 'utf8');
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (error) {
      recordIssue(filePath, `invalid JSON: ${error.message}`);
      continue;
    }
    scanRuleJsonValue(filePath, parsed);
    if (relPath === 'src/rules/inference/dtype.rules.json') {
      checkDtypeRuleFallbacks(filePath, parsed);
    }
  }
}

async function checkCodeRules() {
  for (const rule of CODE_RULES) {
    const filePath = path.join(ROOT, rule.file);
    if (!await pathExists(filePath)) {
      recordIssue(filePath, 'expected source file is missing');
      continue;
    }
    const text = await fs.readFile(filePath, 'utf8');
    const matches = text.match(rule.pattern);
    if (matches?.length) {
      recordIssue(filePath, `${rule.label} appears ${matches.length} time(s)`);
    }
  }
}

async function checkRuntimeConfigs() {
  const runtimeRoot = path.join(ROOT, 'src/config/runtime');
  const files = await collectJsonFiles(runtimeRoot);
  for (const filePath of files) {
    const text = await fs.readFile(filePath, 'utf8');
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (error) {
      recordIssue(filePath, `invalid JSON: ${error.message}`);
      continue;
    }
    scanRuntimeJsonValue(filePath, parsed);
  }
}

async function main() {
  await checkTextFiles();
  await checkRuleJsonFiles();
  await checkCodeRules();
  await checkRuntimeConfigs();

  if (issues.length > 0) {
    console.error('[config:single-source:check] found stale config contract drift:');
    for (const issue of issues) {
      console.error(`- ${issue}`);
    }
    process.exitCode = 1;
    return;
  }

  console.log('[config:single-source:check] config contract single-source checks passed');
}

main().catch((error) => {
  console.error(`[config:single-source:check] ${error.stack || error.message}`);
  process.exitCode = 1;
});
