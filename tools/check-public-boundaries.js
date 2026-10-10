#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import fs from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { assertPackageSourceClosure } from './package-source-closure.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const PACKAGE_CONTENT_LIMITS = Object.freeze({
  // Host composition/evidence extraction and Forge candidate evaluation add five
  // reachable files; compatibility facades replace, not duplicate, implementation.
  // Public Electron signature verification adds its existing declaration to closure.
  // Package-19 inventory: 1755 entries, 2,082,234 packed and 10,765,058 unpacked bytes.
  // The F16 reranker recipe adds 1,560 bytes of pinned helper-kernel declarations.
  // Its generated registry references add 528 unpacked bytes (10,765,586 total).
  // Identical file inventories compress to 2,082,249 bytes with Node 22.22.1/npm 9.2.0
  // and 2,083,864 with CI's Node 22.23.2/npm 10.9.8; payload budgets stay fixed.
  // The embedding F16 recipe, generated reachability, and nullable evidence timings
  // bring this inventory to 1756 files, 2,083,847 packed / 10,780,482 unpacked bytes
  // on Node 22.22.1/npm 9.2.0. Retain the observed cross-npm compression allowance.
  // No repository tools or evaluation fixtures ship; Capsule root remains isolated.
  // The standalone Node host adapter adds one shipped implementation module.
  // Installed-package probe: 1757 files, 2,086,799 packed / 10,791,041 unpacked
  // before the final lifecycle assertions; retain the cross-npm allowance.
  // Application-aware plan selection, including the host snapshot: unchanged
  // 1758-file closure, 2,089,629 packed / 10,802,391 unpacked bytes.
  // Repository onboarding modules stay excluded.
  // Capsule HTTP transport adds six reachable files, not another model engine:
  // 1764 entries, 2,093,537 packed / 10,815,644 unpacked on Node 22/npm 9.
  // Preserve the observed cross-npm compression allowance; core closure stays 46 files.
  // Staged Capsule loading adds five acquisition/policy files. Observed inventory:
  // 1769 entries, 2,096,452 packed / 10,827,703 unpacked on Node 22/npm 9.
  // Retain the cross-npm compression allowance; the isolated core has 49 files.
  // Request-bound adapter policy and exclusive session execution add seven files.
  // Measured npm 9 inventory: 1776 entries, 2,102,329 packed / 10,856,136 unpacked.
  // Explicit LoRA layout policy adds three files. npm 9 measured 1779 entries,
  // 2,103,827 packed / 10,861,748 unpacked bytes; no model weights are shipped.
  // Exporting the existing OPFS backend adds its declaration to the public closure.
  // Public generation contract and shared scalar sampler: measured 1784 files.
  // Evidence: artifacts/generation-contract-2026-09-08/package-audit.json.
  // Versioned incremental streaming adds five shipped files. Exact candidate
  // inventory: artifacts/incremental-streaming-2026-09-12/current-package-audit.json.
  // Ownership contracts and retained import facades add 17 files; no second
  // inference engine. Inventory: artifacts/architecture-consolidation-2026-09-19/package-audit.json.
  // Combined branch integration: 1858 entries, 2,145,728 packed / 11,301,865
  // unpacked bytes on Node 22.22.1/npm 9.2.0. Adds declared GPU selection,
  // rotary source contracts and experimental training; no repository tools.
  // Evidence: artifacts/main-integration-2026-09-19/package-audit.json.
  // The explicit backing-owner port includes its existing declaration in closure.
  // Explicit partition contract JS/types plus extracted model-load orchestration.
  // The minimal Capsule root retains its independent closure check.
  // Resident sessions add ten required JS/declaration files. The installed
  // package smoke measured 1881 entries on Node 22.22.1/npm 9.2.0.
  // Choice scoring and regenerated reachability admit 1,893 package entries.
  // Exact npm 9 inventory: reports/choice-scoring/package-inventory.json.
  // The installed kernel suite requires four captured-operand fixtures already
  // present in 0.6.15 and 0.6.18. No model weights or new runtime modules.
  // Exact 0.6.19 inventory: artifacts/recovery-20261009/package-inventory.json.
  maxEntryCount: 1897,
  // add14e4b: npm 9.2.0 produces 2,104,958 bytes; CI Node 22.23.2/npm 10.9.8
  // produces 2,107,073. Keep the measured cross-toolchain compression allowance.
  // Evidence: reports/capsule-baseline/20260907-release/remote-package-budget-failure.log.
  // Upstream generation constraints add 1,228 source bytes, no files. Measured
  // npm 9: 2,105,323; CI npm 10: 2,107,481. The failure receipt is retained in
  // reports/model-revision-discovery/20260907/remote-merge-budget-failure.log.
  // RNE conversion and reviewed generation constraints keep 1780 entries.
  // Node 22.23.2/npm 10.9.8: 2,107,701 packed; six shipped files add 2,597 bytes.
  // Evidence: artifacts/f16-conversion-rounding-2026-09-07/package-audit.json.
  // Reconciled 810b91fd: 34 shipped files change; ten add 3,571 source bytes.
  // Exact-erf GELU/config binding and static-server changes; closure unchanged.
  // Evidence: artifacts/f16-conversion-rounding-2026-09-07/reconciled-package-audit.json.
  // Opt-in linear-state observation adds 789 bytes in one existing module.
  // Evidence: artifacts/f16-conversion-rounding-2026-09-07/state-probe-package-audit.json.
  // Independent full-prompt reset adds 86 bytes in the existing decode runtime.
  // Evidence: artifacts/f16-conversion-rounding-2026-09-07/prompt-reset-package-audit.json.
  // Measured npm 9: 2,112,518 packed; retain the observed 2,158-byte npm allowance.
  // Before browser-export repair, remote Node 22/npm 10 packed 2,118,944 bytes.
  // The repaired archive measures 2,115,842 on npm 9 and 2,118,939 on Node
  // 22.23.2/npm 10.9.8. Both inventories and the initial failure are retained.
  // The shared-device generation guard adds 223 bytes in the existing GPU module.
  // Node 22.23.2/npm 10.9.8: artifacts/incremental-streaming-2026-09-12/shared-pool-package.json.
  // Consolidated candidate: 2,122,307 packed bytes plus the retained, measured
  // 2,158-byte cross-npm compression allowance.
  // Main integration CI on Node 22.23.2/npm 10.9.8 measures 2,149,817
  // compressed bytes with the same 1858 entries / 11,301,865 unpacked bytes.
  // Preserve exact payload limits; only the measured gzip allowance changes.
  // Evidence: artifacts/main-integration-2026-09-19/remote-ci-initial.json.
  // Shared session-close completion adds 342 unpacked bytes in two existing files.
  // Node 22.23.2/npm 10.9.8: 2,149,978 packed bytes; inventory unchanged.
  // Evidence: artifacts/session-interleaving-2026-09-19/package-audit.json.
  // Adapter/session exclusion: eight shipped files add 5,665 unpacked bytes.
  // Exact Node 22.23.2/npm 10.9.8 archive; no inventory growth.
  // Evidence: artifacts/adapter-session-ownership-2026-09-19/package-audit.json.
  // Main reset forwarding: Node 22.23.2/npm 10.9.8 adds 19 compressed bytes.
  // Evidence: artifacts/release-acceptance-2026-09-19/main-package-audit.json.
  // Incremental hashing plus incoming live-token/cache progress changes: same
  // 1858 files, measured with Node 22.23.2/npm 10.9.8. No tools or fixtures ship.
  // Evidence: artifacts/memory-safe-loading-2026-09-20/package-audit.json.
  // Private block backing and host leases, measured with the same CI toolchain.
  // Evidence: artifacts/memory-safe-loading-2026-09-20/blocks-package-audit.json.
  // Owned-snapshot verification reuse: four existing shipped files change.
  // Evidence: artifacts/startup-verification-2026-09-20/package-audit.json.
  // Bounded streaming acquisition, explicit yielding, and checked reader ports.
  // Evidence: artifacts/bounded-consolidation-2026-09-20/loading-package-audit.json.
  // Direct SHA state reads avoid per-block iteration; one shipped file changes.
  // Evidence: artifacts/bounded-consolidation-2026-09-20/hash-package-audit.json.
  // Owned KV/decode/QKV/RoPE cleanup and allocation-class observations.
  // Evidence: artifacts/bounded-consolidation-2026-09-20/cleanup-package-audit.json.
  // Explicit operation contracts, checked scheduling, and cycle-free model dispatch.
  // Evidence: artifacts/bounded-consolidation-2026-09-20/contracts-package-audit.json.
  // Partition contracts and typed load orchestration: 2,164,412 measured bytes
  // plus the retained 2,158-byte npm compression allowance. Inventory and scope:
  // artifacts/reploid-partitions-2026-09-26/package-audit.json.
  // Final candidate: 2,170,299 bytes on Node 22/npm 9, plus the retained
  // 2,158-byte cross-npm compression allowance.
  // Resident candidate with a shared plan digest measures 2,178,189 packed bytes on npm 9; retain the
  // previously measured 2,158-byte cross-npm compression allowance.
  // Scoring inventory: 2,194,974 packed bytes; preserve the existing measured
  // 2,158-byte cross-npm allowance. No model weights or test operands ship.
  // Those four fixtures and the retained numerical corrections measure
  // 3,339,005 bytes on npm 9; preserve the existing 2,158-byte npm allowance.
  // Head-256 attention refinement: exact payload plus the retained npm allowance.
  // Evidence: artifacts/recovery-20261009/package-inventory-023.json.
  // Resident contract 0.6.27: measured archive plus the existing npm allowance.
  // Evidence: artifacts/partition-contracts-20261010/npm-pack.json.
  maxPackedSize: 3_346_590,
  // Capsule naming changes identifiers and declarations, not the shipped file count.
  // Measured 0.6.0 payload: 2,097,039 packed / 10,835,420 unpacked bytes.
  // The remaining GPU diagnostic label adds three uncompressed bytes.
  // Sequence classification, public OPFS, retention, and device recovery: measured npm 9 archive.
  // GPU logits probe forwarding and failure cleanup add 192 bytes, no files.
  // Evidence: artifacts/f16-conversion-rounding-2026-09-07/probe-package-audit.json.
  // Baseline 6734945b: 1,784 files / 10,885,029 unpacked bytes. Existing stats
  // and benchmark-observation changes preceded installed-consumer work; its
  // application fixtures remain outside the package. The archive inventory is
  // retained in artifacts/installed-consumers-2026-09-12/candidate/npm-pack.json.
  // Main's public reset forwarding method and declaration add exactly 114 bytes
  // in two existing files; all other archive entries are byte-identical.
  // Evidence: artifacts/release-acceptance-2026-09-19/main-package-audit.json.
  // Rig/Run names and compatibility exports: same 1864-file inventory,
  // 2,154,292 packed / 11,334,081 unpacked bytes on Node 22/npm 9.
  // Exact inventory: artifacts/rig-run-naming-2026-09-26/package-audit.json.
  // Loading limits, owned-backing release and digest-preserving hash acceleration:
  // same 1864 files; audited in artifacts/document-search-loading-2026-09-26/package-audit.json.
  // Chunk-local legacy-hash scratch reuse adds 403 bytes in one existing file.
  // Evidence: artifacts/document-search-diagnosis-2026-09-26/package-audit.json.
  // Stateless ModelIR validation/declaration and maintenance README: +320 bytes,
  // unchanged file count. The delivered loader archive remains separately pinned.
  // Evidence: artifacts/document-search-minilm-embedding-2026-09-26/package-audit.json.
  // Final measured candidate; no new package entries.
  // Exact resident candidate: 11,435,846 unpacked bytes; no model weights ship.
  // Scoring contract/qualification and public declarations: 11,504,740 bytes.
  // Baseline drift and this existing-file safety delta are separately audited.
  // Evidence: artifacts/buffer-retirement-2026-10-05/package-audit.json.
  // Exact 0.6.19 payload, including the installed kernel suite's operands.
  // Refined RoPE and attention add shader bytes; the file inventory is unchanged.
  // 0.6.24: unchanged 1897-file inventory; README +2237 and F32 decode +759 bytes.
  // Evidence: artifacts/numerical-20261010/package-audit.json.
  // 0.6.25: the same files; F32 sigmoid gate +249 bytes.
  // Evidence: artifacts/numerical-20261010/package-audit-025.json.
  // 0.6.26: same 1897 files; compensated F32 online attention +4803 bytes.
  // Evidence: artifacts/numerical-20261010/package-audit-026.json.
  // 0.6.27: unchanged inventory; resident declarations, admission and changelog.
  // Evidence: artifacts/partition-contracts-20261010/npm-pack.json.
  maxUnpackedSize: 13_115_542,
});
const REQUIRED_PACKAGE_FILES = Object.freeze([
  'README.md',
  'CHANGELOG.md',
  'LICENSE',
  'assets/doppler.svg',
  'models/catalog.json',
  'src/tooling/command-runner.html',
  'tools/convert-safetensors-node.js',
]);
const FORBIDDEN_PACKAGE_PREFIXES = Object.freeze([
  'src/debug/reference/',
  'src/gpu/kernels/codegen/',
]);
const FORBIDDEN_PACKAGE_FILES = Object.freeze(['assets/doppler-runtime-map.svg']);
const FORBIDDEN_PACKAGE_SUFFIXES = Object.freeze(['.diff', '.py']);

const FILE_RULES = [
  {
    file: 'src/capsule-runtime.js',
    allowed: new Set([
      './version.js',
      './config/generation-contract.js',
      './config/choice-scoring.js',
      './client/runtime/composition-root.js',
      './client/runtime/fetch-capsule-artifact-store.js',
      './client/runtime/capsule-forecast-program.js',
      './client/runtime/capsule-operation-stream.js',
    ]),
    forbidden: [
      'export * from',
      './index.js',
      './client/doppler-api.js',
      './client/provider.js',
      './tooling-exports',
      './converter/',
      './experimental/',
      './models/',
      './inference/',
      './gpu/',
      './storage/',
    ],
  },
  {
    file: 'src/capsule-runtime.d.ts',
    allowed: new Set([
      './version.js',
      './config/generation-contract.js',
      './config/choice-scoring.js',
      './config/capsule.js',
      './config/capsule-operation.js',
      './client/runtime/composition-root.js',
      './client/runtime/fetch-capsule-artifact-store.js',
      './client/runtime/capsule-forecast-program.js',
      './client/runtime/capsule-rerank.js',
      './client/runtime/capsule-operation-stream.js',
      './client/runtime/capsule-operation-executor.js',
    ]),
    forbidden: [
      'export * from',
      './index.js',
      './client/doppler-api.js',
      './client/provider.js',
      './tooling-exports',
      './converter/',
      './experimental/',
      './models/',
      './inference/',
      './gpu/',
      './storage/',
    ],
  },
  {
    file: 'src/index.js',
    allowed: new Set(['./version.js', './client/doppler-api.js', './client/provider.js', './config/generation-contract.js', './client/runtime/capsule-operation-stream.js']),
    forbidden: [
      'export * from',
      './tooling-exports',
      './loader/',
      './loaders/',
      './generation/',
      './experimental/orchestration/',
      './inference/',
      './gpu/',
      './experimental/adapters/',
      './storage/',
      './formats/',
    ],
  },
  {
    file: 'src/index.d.ts',
    allowed: new Set(['./version.js', './client/doppler-api.js', './client/provider.js', './config/generation-contract.js', './client/runtime/capsule-operation-stream.js']),
    forbidden: [
      'export * from',
      './tooling-exports',
      './loader/',
      './loaders/',
      './generation/',
      './experimental/orchestration/',
      './inference/',
      './gpu/',
      './experimental/adapters/',
      './storage/',
      './formats/',
    ],
  },
  {
    file: 'src/index-browser.js',
    allowed: new Set(['./version.js', './client/doppler-api.browser.js']),
    forbidden: [
      'export * from',
      './tooling-exports',
      './loader/',
      './loaders/',
      './generation/',
      './experimental/orchestration/',
      './inference/',
      './gpu/',
      './experimental/adapters/',
      './storage/',
      './config/',
      './formats/',
    ],
  },
  {
    file: 'src/index-browser.d.ts',
    allowed: new Set(['./version.js', './client/doppler-api.browser.js']),
    forbidden: [
      'export * from',
      './tooling-exports',
      './loader/',
      './loaders/',
      './generation/',
      './experimental/orchestration/',
      './inference/',
      './gpu/',
      './experimental/adapters/',
      './storage/',
      './config/',
      './formats/',
    ],
  },
  {
    file: 'src/generation/index.js',
    allowed: new Set(['../inference/pipelines/text.js']),
    forbidden: [
      'export * from',
      '../inference/pipelines/text/config.js',
      '../inference/pipelines/text/init.js',
      '../inference/pipelines/text/model-load.js',
      '../inference/pipelines/structured/',
      '../inference/pipelines/energy-head/',
      '../gpu/',
      '../experimental/adapters/',
    ],
  },
  {
    file: 'src/generation/index.d.ts',
    allowed: new Set([
      '../inference/pipelines/text.js',
      '../inference/pipelines/text/config.js',
      '../inference/pipelines/text/sampling.js',
      '../inference/pipelines/text/lora-types.js',
    ]),
    forbidden: [
      'export * from',
      '../inference/pipelines/text/init.js',
      '../inference/pipelines/text/model-load.js',
      '../inference/pipelines/structured/',
      '../inference/pipelines/energy-head/',
      '../gpu/',
      '../experimental/adapters/',
    ],
  },
];

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

async function readText(relativePath) {
  return fs.readFile(path.join(ROOT_DIR, relativePath), 'utf8');
}

function collectLocalSpecifiers(source) {
  const specifiers = new Set();
  const patterns = [
    /\bimport\s*(?:type\s*)?(?:[^'"]*?\sfrom\s*)?['"]([^'"]+)['"]/g,
    /\bexport\s*(?:type\s*)?(?:[^'"]*?\sfrom\s*)?['"]([^'"]+)['"]/g,
    /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
  ];
  for (const pattern of patterns) {
    pattern.lastIndex = 0;
    for (;;) {
      const match = pattern.exec(source);
      if (!match) break;
      specifiers.add(match[1]);
    }
  }
  return specifiers;
}

function isRelativeSpecifier(specifier) {
  return specifier.startsWith('./') || specifier.startsWith('../');
}

function validateFile(relativePath, rule, source) {
  const specifiers = collectLocalSpecifiers(source);
  for (const specifier of specifiers) {
    if (!isRelativeSpecifier(specifier)) {
      continue;
    }
    const isAllowed = rule.allowed.has(specifier);
    const isForbidden = rule.forbidden.some((token) => specifier.includes(token));
    assert(
      isAllowed,
      `${relativePath} exposes undeclared local specifier "${specifier}". Allowed: ${Array.from(rule.allowed).join(', ')}.`
    );
    assert(
      isAllowed || !isForbidden,
      `${relativePath} exposes forbidden local specifier "${specifier}".`
    );
  }

  for (const token of rule.forbidden) {
    assert(
      !source.includes(token),
      `${relativePath} contains forbidden boundary token "${token}".`
    );
  }
}

function normalizePackageTarget(target) {
  return target.startsWith('./') ? target.slice(2) : target;
}

function assertPackageTargetShape(target, label, options = {}) {
  assert(typeof target === 'string' && target.trim(), `${label} must be a non-empty string target.`);
  if (options.requireNodeExportPrefix) {
    assert(target.startsWith('./'), `${label} must use a package export target beginning with "./".`);
  }
  const normalized = normalizePackageTarget(target);
  assert(!path.isAbsolute(normalized), `${label} must be repo-relative.`);
  assert(!normalized.includes('\\'), `${label} must use forward slashes.`);
  assert(!normalized.split('/').includes('..'), `${label} must not traverse upward.`);
  return normalized;
}

async function assertPackageTargetExists(target, label, options = {}) {
  const normalized = assertPackageTargetShape(target, label, options);
  try {
    await fs.stat(path.join(ROOT_DIR, normalized));
  } catch {
    throw new Error(`${label} target does not exist: ${target}`);
  }
}

async function validatePackageExportTarget(target, label) {
  if (typeof target === 'string') {
    await assertPackageTargetExists(target, label, { requireNodeExportPrefix: true });
    return;
  }
  if (Array.isArray(target)) {
    for (let i = 0; i < target.length; i += 1) {
      await validatePackageExportTarget(target[i], `${label}[${i}]`);
    }
    return;
  }
  assert(target && typeof target === 'object', `${label} must be a string, array, or condition object.`);
  for (const [key, value] of Object.entries(target)) {
    await validatePackageExportTarget(value, `${label}.${key}`);
  }
}

async function validatePackageExportTargets(exportsField) {
  for (const [exportKey, target] of Object.entries(exportsField)) {
    await validatePackageExportTarget(target, `package.json exports ${exportKey}`);
  }
}

async function validatePackageBinTargets(packageJson) {
  const bins = packageJson.bin || {};
  if (typeof bins === 'string') {
    await assertPackageTargetExists(bins, 'package.json bin');
    return;
  }
  assert(bins && typeof bins === 'object' && !Array.isArray(bins), 'package.json bin must be a string or object.');
  for (const [name, target] of Object.entries(bins)) {
    await assertPackageTargetExists(target, `package.json bin ${name}`);
  }
}

function collectPackageTargets(target, output) {
  if (typeof target === 'string') {
    output.add(normalizePackageTarget(target));
    return;
  }
  if (Array.isArray(target)) {
    for (const entry of target) collectPackageTargets(entry, output);
    return;
  }
  if (!target || typeof target !== 'object') return;
  for (const value of Object.values(target)) collectPackageTargets(value, output);
}

function collectRequiredPackageFiles(packageJson) {
  const required = new Set(REQUIRED_PACKAGE_FILES);
  collectPackageTargets(packageJson.exports, required);
  collectPackageTargets(packageJson.bin, required);
  if (typeof packageJson.main === 'string') required.add(normalizePackageTarget(packageJson.main));
  if (typeof packageJson.types === 'string') required.add(normalizePackageTarget(packageJson.types));
  return required;
}

function validatePackedClosure(packedPaths, closure) {
  const requiredPaths = new Set([
    ...closure.runtimeFiles,
    ...closure.typeFiles,
    ...closure.packageResourceFiles,
  ]);
  const missingPaths = [...requiredPaths].filter((requiredPath) => !packedPaths.has(requiredPath));
  assert(
    missingPaths.length === 0,
    `npm package is missing source-closure files:\n${missingPaths.map((file) => `- ${file}`).join('\n')}`
  );

  const extraPaths = [...packedPaths].filter((packedPath) => {
    if (packedPath === 'package.json') return false;
    if (packedPath.endsWith('.js')) return !closure.runtimeFiles.has(packedPath);
    if (packedPath.endsWith('.d.ts')) return !closure.typeFiles.has(packedPath);
    if (
      packedPath.endsWith('.json')
      || packedPath.endsWith('.wgsl')
      || packedPath.endsWith('.html')
    ) {
      return !closure.packageResourceFiles.has(packedPath);
    }
    return false;
  });
  assert(
    extraPaths.length === 0,
    `npm package contains files outside the public source closure:\n${extraPaths.map((file) => `- ${file}`).join('\n')}`
  );
}

function inspectPackedPackage(packageJson, closure) {
  const cacheDir = mkdtempSync(path.join(tmpdir(), 'doppler-npm-package-cache-'));
  try {
    const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
    const result = spawnSync(
      npmCommand,
      ['pack', '--dry-run', '--json', '--ignore-scripts', '--cache', cacheDir],
      {
        cwd: ROOT_DIR,
        encoding: 'utf8',
        maxBuffer: 64 * 1024 * 1024,
      }
    );
    assert(
      result.status === 0,
      `npm pack --dry-run failed:\n${result.stderr || result.stdout || `exit code ${result.status ?? 1}`}`
    );

    let payload;
    try {
      payload = JSON.parse(result.stdout);
    } catch (error) {
      throw new Error(`npm pack --dry-run returned invalid JSON: ${error.message}`);
    }
    assert(Array.isArray(payload) && payload.length === 1, 'npm pack --dry-run must return one package.');

    const packed = payload[0];
    const packedPaths = new Set((packed.files || []).map((entry) => entry.path));
    for (const requiredPath of collectRequiredPackageFiles(packageJson)) {
      assert(packedPaths.has(requiredPath), `npm package is missing required file: ${requiredPath}`);
    }
    validatePackedClosure(packedPaths, closure);

    const forbiddenPaths = [...packedPaths].filter((packedPath) =>
      FORBIDDEN_PACKAGE_FILES.includes(packedPath)
      || FORBIDDEN_PACKAGE_PREFIXES.some((prefix) => packedPath.startsWith(prefix))
      || FORBIDDEN_PACKAGE_SUFFIXES.some((suffix) => packedPath.endsWith(suffix))
    );
    assert(
      forbiddenPaths.length === 0,
      `npm package contains repository-only files:\n${forbiddenPaths.map((file) => `- ${file}`).join('\n')}`
    );
    assert(
      packed.entryCount <= PACKAGE_CONTENT_LIMITS.maxEntryCount,
      `npm package file budget exceeded: ${packed.entryCount} > ${PACKAGE_CONTENT_LIMITS.maxEntryCount}`
    );
    assert(
      packed.size <= PACKAGE_CONTENT_LIMITS.maxPackedSize,
      `npm package packed-size budget exceeded: ${packed.size} > ${PACKAGE_CONTENT_LIMITS.maxPackedSize}`
    );
    assert(
      packed.unpackedSize <= PACKAGE_CONTENT_LIMITS.maxUnpackedSize,
      `npm package unpacked-size budget exceeded: ${packed.unpackedSize} > ${PACKAGE_CONTENT_LIMITS.maxUnpackedSize}`
    );
    return packed;
  } finally {
    rmSync(cacheDir, { recursive: true, force: true });
  }
}

async function validatePackageExports() {
  const packageJson = JSON.parse(await readText('package.json'));
  const exportsField = packageJson.exports || {};

  assert(exportsField['.'], 'package.json exports is missing the root entry.');
  assert(exportsField['./runtime'], 'package.json exports must include ./runtime.');
  assert(exportsField['./compat'], 'package.json exports must include ./compat.');
  assert(exportsField['./provider'], 'package.json exports must include ./provider.');
  assert(!exportsField['./internal'], 'package.json exports must not include ./internal.');
  assert(exportsField['./tooling'], 'package.json exports must include ./tooling.');
  assert(exportsField['./tooling-experimental'], 'package.json exports must include ./tooling-experimental.');
  assert(exportsField['./loaders'], 'package.json exports must include ./loaders.');
  assert(exportsField['./orchestration'], 'package.json exports must include ./orchestration.');
  assert(exportsField['./generation'], 'package.json exports must include ./generation.');
  assert(exportsField['./training'], 'package.json exports must include ./training.');

  const rootExport = exportsField['.'];
  assert(
    rootExport?.import === './src/capsule-runtime.js' && rootExport?.types === './src/capsule-runtime.d.ts',
    'package.json root export must point to src/capsule-runtime.js and src/capsule-runtime.d.ts.'
  );

  const runExport = exportsField['./run'];
  assert(
    runExport?.import === rootExport.import && runExport?.types === rootExport.types,
    'package.json ./run export must alias the Capsule-native root export.'
  );

  const runtimeExport = exportsField['./runtime'];
  assert(
    runtimeExport?.import === rootExport.import && runtimeExport?.types === rootExport.types,
    'package.json ./runtime export must alias the Capsule-native root export.'
  );

  const compatExport = exportsField['./compat'];
  assert(
    compatExport?.import === './src/index.js' && compatExport?.types === './src/index.d.ts',
    'package.json ./compat export must point to src/index.js and src/index.d.ts.'
  );

  const providerExport = exportsField['./provider'];
  assert(
    providerExport?.import === './src/client/provider.js' && providerExport?.types === './src/client/provider.d.ts',
    'package.json ./provider export must point to src/client/provider.js and src/client/provider.d.ts.'
  );

  await validatePackageExportTargets(exportsField);
  await validatePackageBinTargets(packageJson);
  return packageJson;
}

async function main() {
  const packageJson = await validatePackageExports();
  const packageClosure = await assertPackageSourceClosure(packageJson);

  for (const rule of FILE_RULES) {
    const source = await readText(rule.file);
    validateFile(rule.file, rule, source);
  }

  const packed = inspectPackedPackage(packageJson, packageClosure);
  console.log(
    `public boundary check passed (npm package: ${packed.entryCount} files, `
    + `${packed.size} bytes packed, ${packed.unpackedSize} bytes unpacked)`
  );
}

try {
  await main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
