#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

import { DEFAULT_EXTERNAL_MODELS_ROOT } from './hf-registry-utils.js';

const DEFAULT_VOLUME_ROOT = DEFAULT_EXTERNAL_MODELS_ROOT;
const DEFAULT_RDRR_ROOT = path.join(DEFAULT_VOLUME_ROOT, 'rdrr');
const DEFAULT_JSON_OUTPUT = path.join(DEFAULT_VOLUME_ROOT, 'VOLUME_INDEX.json');
const DEFAULT_MD_OUTPUT = path.join(DEFAULT_VOLUME_ROOT, 'VOLUME_INDEX.md');

function parseArgs(argv) {
  const out = {
    check: false,
    volumeRoot: path.resolve(DEFAULT_VOLUME_ROOT),
    rdrrRoot: path.resolve(DEFAULT_RDRR_ROOT),
    jsonOutput: path.resolve(DEFAULT_JSON_OUTPUT),
    mdOutput: path.resolve(DEFAULT_MD_OUTPUT),
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--check') {
      out.check = true;
      continue;
    }
    if (arg === '--volume-root') {
      const value = String(argv[i + 1] || '').trim();
      if (!value) throw new Error('Missing value for --volume-root');
      out.volumeRoot = path.resolve(value);
      i += 1;
      continue;
    }
    if (arg === '--rdrr-root') {
      const value = String(argv[i + 1] || '').trim();
      if (!value) throw new Error('Missing value for --rdrr-root');
      out.rdrrRoot = path.resolve(value);
      i += 1;
      continue;
    }
    if (arg === '--json-output') {
      const value = String(argv[i + 1] || '').trim();
      if (!value) throw new Error('Missing value for --json-output');
      out.jsonOutput = path.resolve(value);
      i += 1;
      continue;
    }
    if (arg === '--md-output') {
      const value = String(argv[i + 1] || '').trim();
      if (!value) throw new Error('Missing value for --md-output');
      out.mdOutput = path.resolve(value);
      i += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  return out;
}

function toPosix(filePath) {
  return filePath.replace(/\\/g, '/');
}

function escapeCell(value) {
  return String(value ?? '').replace(/\|/g, '\\|').replace(/\n/g, ' ');
}

function formatGiB(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0.00';
  return (bytes / (1024 ** 3)).toFixed(2);
}

async function readJsonIfExists(filePath) {
  try {
    const raw = await fs.readFile(filePath, 'utf8');
    return JSON.parse(raw);
  } catch (error) {
    if (error?.code === 'ENOENT') return null;
    throw error;
  }
}

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function collectManifestPaths(rootDir) {
  const out = [];
  if (!(await exists(rootDir))) {
    return out;
  }
  const stack = [rootDir];
  while (stack.length > 0) {
    const current = stack.pop();
    const entries = await fs.readdir(current, { withFileTypes: true });
    const ordered = entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of ordered) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
        continue;
      }
      if (entry.isFile() && entry.name === 'manifest.json') {
        out.push(fullPath);
      }
    }
  }
  out.sort((a, b) => a.localeCompare(b));
  return out;
}

function toIsoDate(isoDate) {
  if (typeof isoDate !== 'string' || !isoDate.trim()) return 'unknown';
  const parsed = new Date(isoDate);
  if (Number.isNaN(parsed.getTime())) return 'unknown';
  return parsed.toISOString().slice(0, 10);
}

function normalizeToken(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function buildMarkdown(payload) {
  const lines = [];
  lines.push('# External RDRR Index');
  lines.push('');
  lines.push(`Generated: ${payload.generatedAt}`);
  lines.push(`Volume root: \`${payload.volumeRoot}\``);
  lines.push(`RDRR root: \`${payload.rdrrRoot}\``);
  lines.push('');
  lines.push(`- Source models tracked: ${payload.summary.sourceModelCount}`);
  lines.push(`- RDRR variants tracked: ${payload.summary.variantCount}`);
  lines.push('');
  lines.push('To map one safetensors source to multiple quantized variants, place an `origin.json` next to each `manifest.json` with `sourceRepo`, `sourceFormat`, and optional `variant`.');
  lines.push('');
  lines.push('## Source Models');
  lines.push('');
  lines.push('| Source model | Formats | Variant count | RDRR model IDs |');
  lines.push('| --- | --- | ---: | --- |');
  for (const source of payload.sourceModels) {
    const ids = source.variants.map((variant) => variant.rdrrModelId).join(', ');
    lines.push(
      `| ${escapeCell(source.sourceModel)} | ${escapeCell(source.sourceFormats.join(', '))} | ${source.variantCount} | ${escapeCell(ids)} |`
    );
  }
  lines.push('');
  lines.push('## Variants');
  lines.push('');
  lines.push('| Source model | RDRR model ID | Variant | Quantization | Converted | Size (GiB) | Shards | Path |');
  lines.push('| --- | --- | --- | --- | --- | ---: | ---: | --- |');
  for (const source of payload.sourceModels) {
    for (const variant of source.variants) {
      lines.push(
        `| ${escapeCell(source.sourceModel)} | ${escapeCell(variant.rdrrModelId)} | ${escapeCell(variant.variant || 'unknown')} | ${escapeCell(variant.quantization || 'unknown')} | ${escapeCell(toIsoDate(variant.convertedAt))} | ${escapeCell(formatGiB(variant.totalSizeBytes))} | ${variant.shardCount} | ${escapeCell(variant.pathRelativeToVolume)} |`
      );
    }
  }
  lines.push('');
  return `${lines.join('\n')}\n`;
}

async function buildIndex(args, generatedAt = new Date().toISOString()) {
  const manifestPaths = await collectManifestPaths(args.rdrrRoot);
  const variants = [];

  for (const manifestPath of manifestPaths) {
    const manifest = await readJsonIfExists(manifestPath);
    if (!manifest || typeof manifest !== 'object') continue;

    const manifestDir = path.dirname(manifestPath);
    const originPath = path.join(manifestDir, 'origin.json');
    const origin = await readJsonIfExists(originPath);
    const modelId = normalizeToken(manifest.modelId) || path.basename(manifestDir);
    const sourceRepo = normalizeToken(origin?.sourceRepo || origin?.sourceModel || manifest?.metadata?.sourceRepo);
    const sourceFormat = normalizeToken(origin?.sourceFormat || manifest?.metadata?.sourceFormat || 'unknown');
    const sourceRevision = normalizeToken(origin?.sourceRevision || manifest?.metadata?.sourceRevision);
    const sourceModel = sourceRepo || normalizeToken(manifest?.metadata?.sourceModel);
    const variantName = normalizeToken(origin?.variant || manifest?.metadata?.variant);
    if (!sourceModel) {
      throw new Error(
        `Missing explicit sourceModel/sourceRepo metadata for ${toPosix(path.relative(args.volumeRoot, manifestPath))}. ` +
        'Add origin.json or manifest.metadata.sourceModel/sourceRepo.'
      );
    }
    if (!variantName) {
      throw new Error(
        `Missing explicit variant metadata for ${toPosix(path.relative(args.volumeRoot, manifestPath))}. ` +
        'Add origin.json.variant or manifest.metadata.variant.'
      );
    }
    const shardCount = Array.isArray(manifest.shards) ? manifest.shards.length : 0;
    const totalSizeBytes = Number.isFinite(manifest.totalSize) ? manifest.totalSize : 0;

    variants.push({
      sourceModel,
      sourceFormat: sourceFormat || 'unknown',
      sourceRevision: sourceRevision || null,
      rdrrModelId: modelId,
      variant: variantName || null,
      quantization: normalizeToken(manifest.quantization) || null,
      convertedAt: normalizeToken(manifest?.metadata?.convertedAt) || null,
      totalSizeBytes,
      shardCount,
      manifestPath: toPosix(manifestPath),
      pathRelativeToVolume: toPosix(path.relative(args.volumeRoot, manifestDir)),
      pathRelativeToRdrrRoot: toPosix(path.relative(args.rdrrRoot, manifestDir)),
      hasOrigin: Boolean(origin),
    });
  }

  variants.sort((a, b) => {
    if (a.sourceModel !== b.sourceModel) return a.sourceModel.localeCompare(b.sourceModel);
    return a.rdrrModelId.localeCompare(b.rdrrModelId);
  });

  const grouped = new Map();
  for (const variant of variants) {
    if (!grouped.has(variant.sourceModel)) {
      grouped.set(variant.sourceModel, {
        sourceModel: variant.sourceModel,
        sourceFormats: new Set(),
        variants: [],
      });
    }
    const bucket = grouped.get(variant.sourceModel);
    bucket.sourceFormats.add(variant.sourceFormat || 'unknown');
    bucket.variants.push({
      rdrrModelId: variant.rdrrModelId,
      variant: variant.variant,
      quantization: variant.quantization,
      convertedAt: variant.convertedAt,
      totalSizeBytes: variant.totalSizeBytes,
      shardCount: variant.shardCount,
      sourceFormat: variant.sourceFormat,
      sourceRevision: variant.sourceRevision,
      pathRelativeToVolume: variant.pathRelativeToVolume,
      pathRelativeToRdrrRoot: variant.pathRelativeToRdrrRoot,
      manifestPath: variant.manifestPath,
      hasOrigin: variant.hasOrigin,
    });
  }

  const sourceModels = [...grouped.values()]
    .sort((a, b) => a.sourceModel.localeCompare(b.sourceModel))
    .map((entry) => ({
      sourceModel: entry.sourceModel,
      sourceFormats: [...entry.sourceFormats].sort((a, b) => a.localeCompare(b)),
      variantCount: entry.variants.length,
      variants: entry.variants.sort((a, b) => a.rdrrModelId.localeCompare(b.rdrrModelId)),
    }));

  const payload = {
    schemaVersion: 1,
    generatedAt,
    volumeRoot: toPosix(args.volumeRoot),
    rdrrRoot: toPosix(args.rdrrRoot),
    summary: {
      sourceModelCount: sourceModels.length,
      variantCount: variants.length,
    },
    sourceModels,
  };

  return {
    json: `${JSON.stringify(payload, null, 2)}\n`,
    markdown: buildMarkdown(payload),
  };
}

async function readFileIfExists(filePath) {
  try {
    return await fs.readFile(filePath, 'utf8');
  } catch (error) {
    if (error?.code === 'ENOENT') return null;
    throw error;
  }
}

function normalizeJsonForCheck(raw) {
  if (typeof raw !== 'string' || !raw.trim()) return null;
  const parsed = JSON.parse(raw);
  if (parsed && typeof parsed === 'object') {
    delete parsed.generatedAt;
  }
  return JSON.stringify(parsed);
}

function normalizeMarkdownForCheck(raw) {
  if (typeof raw !== 'string') return null;
  return raw.replace(/^Generated: .+$/m, 'Generated: <normalized>');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.check) {
    const outputs = await buildIndex(args, '<normalized>');
    const [currentJson, currentMd] = await Promise.all([
      readFileIfExists(args.jsonOutput),
      readFileIfExists(args.mdOutput),
    ]);
    const jsonMatches =
      normalizeJsonForCheck(currentJson) === normalizeJsonForCheck(outputs.json);
    const mdMatches =
      normalizeMarkdownForCheck(currentMd) === normalizeMarkdownForCheck(outputs.markdown);
    if (!jsonMatches || !mdMatches) {
      throw new Error(
        `External RDRR index is out of date. Run: node tools/sync-external-rdrr-index.js ` +
        `--volume-root ${args.volumeRoot}`
      );
    }
    console.error(
      `[external-rdrr-index] up to date (${args.jsonOutput}, ${args.mdOutput})`
    );
    return;
  }

  const outputs = await buildIndex(args);

  await fs.mkdir(path.dirname(args.jsonOutput), { recursive: true });
  await fs.mkdir(path.dirname(args.mdOutput), { recursive: true });
  await Promise.all([
    fs.writeFile(args.jsonOutput, outputs.json, 'utf8'),
    fs.writeFile(args.mdOutput, outputs.markdown, 'utf8'),
  ]);

  console.error(
    `[external-rdrr-index] wrote ${args.jsonOutput} and ${args.mdOutput}`
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(`[external-rdrr-index] ${error.message}`);
    process.exit(1);
  });
}

export {
  buildIndex,
};
