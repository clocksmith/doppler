#!/usr/bin/env node


import { readFile } from 'fs/promises';
import { resolve, join } from 'path';
import { log } from './index.js';
import { loadConfig } from '../../cli/config/index.js';

const PROJECT_ROOT = resolve(new URL('.', import.meta.url).pathname, '..', '..');

function parseArgs(argv) {
  const opts = { config: null, help: false };
  let i = 0;
  while (i < argv.length) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      opts.help = true;
      i++;
      continue;
    }
    if (arg === '--config' || arg === '-c') {
      opts.config = argv[i + 1] || null;
      i += 2;
      continue;
    }
    if (!arg.startsWith('-') && !opts.config) {
      opts.config = arg;
      i++;
      continue;
    }
    console.error(`Unknown argument: ${arg}`);
    opts.help = true;
    break;
  }
  return opts;
}

function assertString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string`);
  }
}

async function diagnoseModel(modelPath) {
  const result = {
    modelPath,
    manifestValid: false,
    hasKernelPath: false,
    issues: [],
    recommendations: [],
  };

  try {
    // Read manifest
    const manifestPath = join(modelPath, 'manifest.json');
    const manifestText = await readFile(manifestPath, 'utf-8');
    const manifest = JSON.parse(manifestText);
    result.manifestValid = true;

    // Check Q4K layout - now in quantizationInfo.layout
    // 'row' = fused kernel (fast), 'col' = dequant fallback (slow)
    const q4kLayout = manifest.quantizationInfo?.layout ?? null;
    result.q4kLayout = q4kLayout;

    if (!q4kLayout) {
      result.issues.push('Warning: Missing quantizationInfo.layout field');
      result.recommendations.push('Re-convert model to add Q4K layout metadata');
    } else if (q4kLayout === 'col') {
      result.issues.push('Warning: Q4K layout is "col" (dequant fallback, loses Q4K benefits)');
      result.recommendations.push('Re-convert model with converter.quantization.q4kLayout="row" for fused Q4K (faster, smaller)');
    } else if (q4kLayout === 'row') {
      log.info('KernelDiag', 'OK Q4K layout: row (fused kernel, optimal)');
    } else {
      log.info('KernelDiag', `Q4K layout: ${q4kLayout}`);
    }

    // Check kernel path override (optional)
    const kernelPath = manifest.optimizations?.kernelPath ?? manifest.inference?.defaultKernelPath;
    result.kernelPath = kernelPath;
    result.hasKernelPath = kernelPath != null;

    if (!kernelPath) {
      result.recommendations.push('No kernelPath override found; kernel selection will be automatic.');
    } else {
      log.info('KernelDiag', `OK kernelPath override present: ${JSON.stringify(kernelPath)}`);
    }

    // Check quantization type
    if (manifest.quantization !== 'Q4_K_M') {
      result.issues.push(`Info: Quantization is "${manifest.quantization}" (Q4K hints apply to Q4_K_M only)`);
    }

  } catch (error) {
    result.issues.push(`Error reading manifest: ${error.message}`);
  }

  return result;
}

async function printDiagnostics(result) {
  log.info('KernelDiag', '='.repeat(70));
  log.info('KernelDiag', 'DOPPLER Kernel Diagnostics');
  log.info('KernelDiag', '='.repeat(70));
  log.info('KernelDiag', `Model: ${result.modelPath}`);
  log.info('KernelDiag', `Manifest: ${result.manifestValid ? 'OK Valid' : 'ERROR Invalid'}`);

  if (result.q4kLayout) {
    const isOptimal = result.q4kLayout === 'row';
    const layoutStatus = isOptimal ? 'OK' : 'WARNING';
    log.info('KernelDiag', `Q4K Layout: ${layoutStatus} ${result.q4kLayout}`);
  }

  if (result.hasKernelPath) {
    log.info('KernelDiag', 'Kernel Path: OK Present');
  } else {
    log.info('KernelDiag', 'Kernel Path: auto (no override)');
  }

  if (result.issues.length > 0) {
    log.warn('KernelDiag', 'Issues Found:');
    result.issues.forEach(issue => log.warn('KernelDiag', `  ${issue}`));
  }

  if (result.recommendations.length > 0) {
    log.info('KernelDiag', 'Recommendations:');
    result.recommendations.forEach(rec => log.info('KernelDiag', `  ${rec}`));
  }

  if (result.issues.length === 0) {
    log.info('KernelDiag', 'Model is optimally configured!');
    log.info('KernelDiag', 'Expected Performance:');
    log.info('KernelDiag', '  - Gemma 3 1B: ~8 tok/s (Apple M3)');
    log.info('KernelDiag', '  - Column-wise layout: +14% vs flat');
    log.info('KernelDiag', 'Run benchmark via config (cli.command=bench, cli.suite=inference).');
  }

  log.info('KernelDiag', '='.repeat(70));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    log.info('KernelDiag', 'Usage: doppler --config <ref>');
    log.info('KernelDiag', 'Config requirements:');
    log.info('KernelDiag', '  - model (string, required)');
    log.info('KernelDiag', '  - tools.diagnoseKernels.modelPath (string, optional; overrides models/<model>)');
    log.info('KernelDiag', 'Checks:');
    log.info('KernelDiag', '  - Manifest validity');
    log.info('KernelDiag', '  - Q4K layout configuration (row=fused, col=dequant)');
    log.info('KernelDiag', '  - Kernel path override (optional)');
    log.info('KernelDiag', '  - Expected vs actual configuration');
    process.exit(0);
  }

  if (!args.config) {
    console.error('Error: --config is required');
    process.exit(1);
  }

  const loadedConfig = await loadConfig(args.config);
  const raw = loadedConfig.raw ?? {};
  assertString(raw.model, 'model');
  const tool = raw.tools?.diagnoseKernels ?? {};
  if (tool.modelPath !== undefined) {
    assertString(tool.modelPath, 'tools.diagnoseKernels.modelPath');
  }
  const modelPath = tool.modelPath
    ? resolve(PROJECT_ROOT, tool.modelPath)
    : join(PROJECT_ROOT, 'models', raw.model);
  const result = await diagnoseModel(modelPath);
  await printDiagnostics(result);

  // Exit with error code if issues found
  process.exit(result.issues.length > 0 ? 1 : 0);
}

main().catch(error => {
  log.error('KernelDiag', `Fatal error: ${error.message}`);
  process.exit(1);
});
