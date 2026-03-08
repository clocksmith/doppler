import { existsSync, readFileSync, writeFileSync } from 'fs';
import { readdir } from 'fs/promises';
import { join, relative } from 'path';

export const ROOT = new URL('..', import.meta.url).pathname;

const SRC_OWNER_RULES = [
  { owner: 'A', agent: 'config-owner', test: f => f === 'src/inference/pipelines/text/execution-v0.js' },
  { owner: 'A', agent: 'config-owner', test: f => f === 'src/inference/pipelines/text/model-load.js' },
  { owner: 'C', agent: 'surface-owner', test: f => f === 'src/inference/browser-harness.js' },
  {
    owner: 'A',
    agent: 'config-owner',
    test: f => f.startsWith('src/config/') || f.startsWith('src/rules/') || f.startsWith('src/converter/'),
  },
  {
    owner: 'B',
    agent: 'runtime-owner',
    test: f =>
      f.startsWith('src/inference/') || f.startsWith('src/gpu/') || f.startsWith('src/loader/') ||
      f.startsWith('src/memory/') || f.startsWith('src/formats/') || f.startsWith('src/training/') ||
      f.startsWith('src/generation/') || f.startsWith('src/diffusion/') || f.startsWith('src/energy/'),
  },
  { owner: 'C', agent: 'surface-owner', test: f => f.startsWith('src/') },
];

const TOOLS_A_FILES = new Set([
  'tools/check-contract-artifacts.js',
  'tools/check-hf-registry.js',
  'tools/convert-safetensors-node.js',
  'tools/hf-registry-utils.js',
  'tools/lean-execution-contract-config-sweep.js',
  'tools/lean-execution-contract-sweep.js',
  'tools/lean-execution-contract.js',
  'tools/publish-hf-registry-model.js',
  'tools/refresh-converted-manifest.js',
  'tools/summarize-conversion-reports.js',
  'tools/sync-external-rdrr-index.js',
  'tools/verify-training-provenance.mjs',
  'tools/verify-training-workload-packs.mjs',
]);

const TOOLS_B_FILES = new Set([
  'tools/ci-diffusion-contract-gates.mjs',
  'tools/ci-training-contract-gates.mjs',
  'tools/compare-engines.js',
  'tools/compare-ul-runs.mjs',
  'tools/distillation.js',
  'tools/emit-training-contract-delta.mjs',
  'tools/generate-wgsl.js',
  'tools/lora.js',
  'tools/node-test-runtime-setup.mjs',
  'tools/p2p-delivery-observability.mjs',
  'tools/p2p-resilience-drill.mjs',
  'tools/run-distill-bench.mjs',
  'tools/run-node-coverage.mjs',
  'tools/run-node-tests.mjs',
  'tools/run-ul-bench.mjs',
  'tools/vendor-bench.js',
  'tools/wgsl-variant-generator.js',
]);

const TOOLS_OWNER_RULES = [
  {
    owner: 'A',
    agent: 'config-owner',
    test: f =>
      f.startsWith('tools/configs/') ||
      f.startsWith('tools/policies/') ||
      TOOLS_A_FILES.has(f),
  },
  {
    owner: 'B',
    agent: 'runtime-owner',
    test: f =>
      TOOLS_B_FILES.has(f) ||
      f === 'tools/configs/wgsl-patch-variants.js' ||
      f === 'tools/configs/wgsl-variants.js',
  },
  { owner: 'C', agent: 'surface-owner', test: f => f.startsWith('tools/') },
];

export const AUDIT_SCOPES = Object.freeze({
  src: {
    name: 'src',
    rootDir: join(ROOT, 'src'),
    auditDir: join(ROOT, 'reports/review/src-audit'),
    matchFile: name => /\.(js|wgsl)$/.test(name) && !name.endsWith('.d.js'),
    ownerRules: SRC_OWNER_RULES,
  },
  tools: {
    name: 'tools',
    rootDir: join(ROOT, 'tools'),
    auditDir: join(ROOT, 'reports/review/tools-audit'),
    matchFile: () => true,
    ownerRules: TOOLS_OWNER_RULES,
  },
});

export function getScopeConfig(scopeName = 'src') {
  const scope = AUDIT_SCOPES[scopeName];
  if (!scope) {
    throw new Error(`Unknown scope "${scopeName}". Valid scopes: ${Object.keys(AUDIT_SCOPES).join(', ')}`);
  }
  return scope;
}

export function assignOwner(scopeName, relPath) {
  const scope = getScopeConfig(scopeName);
  for (const rule of scope.ownerRules) {
    if (rule.test(relPath)) return { owner: rule.owner, agent: rule.agent };
  }
  return { owner: 'C', agent: 'surface-owner' };
}

export async function walkScope(scopeName) {
  const scope = getScopeConfig(scopeName);
  return walkDirectory(scope.rootDir, scope.matchFile);
}

async function walkDirectory(dir, matchFile) {
  const files = [];
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await walkDirectory(full, matchFile));
      continue;
    }
    if (entry.isFile() && matchFile(entry.name, full)) {
      files.push(full);
    }
  }
  return files;
}

export function getAuditPaths(scopeName) {
  const scope = getScopeConfig(scopeName);
  return {
    scope,
    auditDir: scope.auditDir,
    eventsFile: join(scope.auditDir, 'events.jsonl'),
    latestFile: join(scope.auditDir, 'latest.jsonl'),
  };
}

export function validateEventSequence(events, label = 'events') {
  let previousSeq = 0;
  const seen = new Set();
  for (const event of events) {
    if (!Number.isInteger(event?.seq) || event.seq < 1) {
      throw new Error(`${label} contains an invalid seq value.`);
    }
    if (seen.has(event.seq)) {
      throw new Error(`${label} contains duplicate seq=${event.seq}.`);
    }
    if (event.seq <= previousSeq) {
      throw new Error(`${label} must be strictly increasing by file order; saw ${event.seq} after ${previousSeq}.`);
    }
    seen.add(event.seq);
    previousSeq = event.seq;
  }
}

export function readJsonl(file) {
  if (!existsSync(file)) return [];
  return readFileSync(file, 'utf8')
    .split('\n')
    .filter(line => line.trim())
    .map(line => JSON.parse(line));
}

export function writeJsonl(file, rows) {
  const content = rows.map(row => JSON.stringify(row)).join('\n');
  writeFileSync(file, content ? content + '\n' : '');
}

export function toRelativePath(absPath) {
  return relative(ROOT, absPath);
}
