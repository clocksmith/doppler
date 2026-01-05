#!/usr/bin/env node
/**
 * Update manifest settings without touching shards.
 *
 * Safe edits (default):
 * - optimizations.kernelPlan
 *
 * Unsafe edits (require --allow-unsafe):
 * - config.q4kLayout
 * - defaultWeightLayout
 */

import { readFile, writeFile, stat } from 'fs/promises';
import { resolve, join } from 'path';

interface UpdateOptions {
  input: string | null;
  kernelPlan: Record<string, unknown> | null;
  kernelPlanMode: 'patch' | 'replace' | null;
  kernelPlanStrict: boolean | null;
  q4kStrategy: string | null;
  matmulVariant: string | null;
  attentionVariant: string | null;
  attentionPrefill: string | null;
  attentionDecode: string | null;
  clearKernelPlan: boolean;
  q4kLayout: string | null;
  defaultWeightLayout: string | null;
  allowUnsafe: boolean;
  dryRun: boolean;
  help: boolean;
}

function parseArgs(args: string[]): UpdateOptions {
  const options: UpdateOptions = {
    input: null,
    kernelPlan: null,
    kernelPlanMode: null,
    kernelPlanStrict: null,
    q4kStrategy: null,
    matmulVariant: null,
    attentionVariant: null,
    attentionPrefill: null,
    attentionDecode: null,
    clearKernelPlan: false,
    q4kLayout: null,
    defaultWeightLayout: null,
    allowUnsafe: false,
    dryRun: false,
    help: false,
  };

  let i = 0;
  while (i < args.length) {
    const arg = args[i];
    switch (arg) {
      case '--help':
      case '-h':
        options.help = true;
        break;
      case '--kernel-plan': {
        const raw = args[++i] || '';
        try {
          const parsed = JSON.parse(raw);
          if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            throw new Error('kernel plan must be a JSON object');
          }
          options.kernelPlan = parsed as Record<string, unknown>;
        } catch (err) {
          throw new Error(`Failed to parse --kernel-plan JSON: ${(err as Error).message}`);
        }
        break;
      }
      case '--kernel-plan-mode': {
        const mode = (args[++i] || '').toLowerCase();
        if (mode === 'patch' || mode === 'replace') {
          options.kernelPlanMode = mode;
        }
        break;
      }
      case '--kernel-plan-strict':
        options.kernelPlanStrict = true;
        break;
      case '--kernel-plan-lax':
        options.kernelPlanStrict = false;
        break;
      case '--q4k-strategy':
        options.q4kStrategy = args[++i] || null;
        break;
      case '--matmul-variant':
        options.matmulVariant = args[++i] || null;
        break;
      case '--attention-variant':
        options.attentionVariant = args[++i] || null;
        break;
      case '--attention-prefill':
        options.attentionPrefill = args[++i] || null;
        break;
      case '--attention-decode':
        options.attentionDecode = args[++i] || null;
        break;
      case '--clear-kernel-plan':
        options.clearKernelPlan = true;
        break;
      case '--q4k-layout':
        options.q4kLayout = args[++i] || null;
        break;
      case '--default-weight-layout':
        options.defaultWeightLayout = args[++i] || null;
        break;
      case '--allow-unsafe':
        options.allowUnsafe = true;
        break;
      case '--dry-run':
        options.dryRun = true;
        break;
      default:
        if (!arg.startsWith('-') && !options.input) {
          options.input = arg;
        }
        break;
    }
    i++;
  }

  return options;
}

function printHelp(): void {
  console.log(`
Update DOPPLER manifest settings (no shard changes).

Usage:
  node update-manifest.js <model-dir|manifest.json> [options]

Safe options:
  --kernel-plan <json>
  --kernel-plan-mode <patch|replace>
  --kernel-plan-strict
  --kernel-plan-lax
  --q4k-strategy <auto|fused_q4k|dequant_f16|dequant_f32>
  --matmul-variant <variant>
  --attention-variant <tier|variant>
  --attention-prefill <tier|variant>
  --attention-decode <tier|variant>
  --clear-kernel-plan
  --dry-run

Unsafe options (require --allow-unsafe):
  --q4k-layout <flat|row_wise|column_wise>
  --default-weight-layout <row|column>

Examples:
  node update-manifest.js ./models/gemma-1b-q4-row --q4k-strategy fused_q4k
  node update-manifest.js ./models/gemma-1b-q4-row --attention-decode streaming
  node update-manifest.js ./models/gemma-1b-q4-row --kernel-plan '{"q4kStrategy":"dequant_f16","variants":{"attention":{"prefill":"tiled_small"}}}'
`);
}

async function resolveManifestPath(input: string): Promise<string> {
  const resolved = resolve(input);
  const stats = await stat(resolved);
  if (stats.isDirectory()) {
    return join(resolved, 'manifest.json');
  }
  return resolved;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    process.exit(0);
  }

  if (!options.input) {
    console.error('Error: model directory or manifest.json path is required.');
    process.exit(1);
  }

  const manifestPath = await resolveManifestPath(options.input);
  const manifestJson = await readFile(manifestPath, 'utf-8');
  const manifest = JSON.parse(manifestJson);
  let changed = false;

  if (options.clearKernelPlan) {
    if (manifest.optimizations?.kernelPlan) {
      delete manifest.optimizations.kernelPlan;
      changed = true;
    }
  }

  const hasKernelPlanUpdates =
    options.kernelPlan ||
    options.kernelPlanMode ||
    options.kernelPlanStrict !== null ||
    options.q4kStrategy ||
    options.matmulVariant ||
    options.attentionVariant ||
    options.attentionPrefill ||
    options.attentionDecode;

  if (hasKernelPlanUpdates) {
    manifest.optimizations = manifest.optimizations || {};

    const existingPlan = manifest.optimizations.kernelPlan || {};
    const kernelPlan = options.kernelPlan ? { ...options.kernelPlan } : { ...existingPlan };

    if (options.kernelPlanMode) {
      kernelPlan.mode = options.kernelPlanMode;
    }
    if (options.kernelPlanStrict !== null) {
      kernelPlan.strict = options.kernelPlanStrict;
    }
    if (options.q4kStrategy) {
      kernelPlan.q4kStrategy = options.q4kStrategy;
    }

    if (options.matmulVariant || options.attentionVariant || options.attentionPrefill || options.attentionDecode) {
      const variants = { ...(kernelPlan.variants as Record<string, any> | undefined) };

      if (options.matmulVariant) {
        const matmul = { ...(variants.matmul ?? {}) };
        matmul.default = options.matmulVariant;
        variants.matmul = matmul;
      }

      if (options.attentionVariant || options.attentionPrefill || options.attentionDecode) {
        const attention = { ...(variants.attention ?? {}) };
        if (options.attentionVariant) attention.default = options.attentionVariant;
        if (options.attentionPrefill) attention.prefill = options.attentionPrefill;
        if (options.attentionDecode) attention.decode = options.attentionDecode;
        variants.attention = attention;
      }

      kernelPlan.variants = variants;
    }

    manifest.optimizations.kernelPlan = kernelPlan;
    changed = true;
  }

  if (options.q4kLayout) {
    if (!options.allowUnsafe) {
      console.warn('Skipping --q4k-layout (use --allow-unsafe to force).');
    } else {
      manifest.config = manifest.config || {};
      manifest.config.q4kLayout = options.q4kLayout;
      changed = true;
    }
  }

  if (options.defaultWeightLayout) {
    if (!options.allowUnsafe) {
      console.warn('Skipping --default-weight-layout (use --allow-unsafe to force).');
    } else {
      manifest.defaultWeightLayout = options.defaultWeightLayout;
      changed = true;
    }
  }

  if (!changed) {
    console.log('No changes requested.');
    return;
  }

  if (options.dryRun) {
    console.log(`Dry run: changes applied in memory for ${manifestPath}`);
    return;
  }

  await writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`Updated manifest: ${manifestPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
