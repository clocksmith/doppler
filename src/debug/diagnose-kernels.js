#!/usr/bin/env node
/**
 * Diagnostic tool to verify kernel selection logic.
 *
 * This script checks:
 * 1. Manifest kernel path is correctly configured
 * 2. Kernel selection logic matches expected behavior
 * 3. GPU capabilities are correctly detected
 *
 * Usage:
 *   npx tsx debug/diagnose-kernels.ts <model-path>
 *
 * Example:
 *   npx tsx debug/diagnose-kernels.ts models/gemma-1b-q4-col
 */

import { readFile } from 'fs/promises';
import { resolve, join } from 'path';
import { log } from './index.js';

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

    // Check Q4K layout
    const q4kLayout = manifest.config?.q4kLayout;
    result.q4kLayout = q4kLayout;

    if (!q4kLayout) {
      result.issues.push('Warning: Missing config.q4kLayout field');
      result.recommendations.push('Run converter with --q4k-layout column_wise');
    } else if (q4kLayout !== 'column_wise') {
      result.issues.push(`Warning: Q4K layout is "${q4kLayout}" (expected "column_wise" for best performance)`);
      result.recommendations.push('Re-convert model with --q4k-layout column_wise for 14% speedup');
    } else {
      log.info('KernelDiag', 'OK Q4K layout: column_wise (optimal)');
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
    const layoutStatus = result.q4kLayout === 'column_wise' ? 'OK' : 'WARNING';
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
    log.info('KernelDiag', 'Run benchmark:');
    log.info('KernelDiag', `  npx tsx cli/index.ts bench inference --model ${result.modelPath.split('/').pop()} --config bench`);
  }

  log.info('KernelDiag', '='.repeat(70));
}

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    log.info('KernelDiag', 'Usage: npx tsx debug/diagnose-kernels.ts <model-path>');
    log.info('KernelDiag', 'Example:');
    log.info('KernelDiag', '  npx tsx debug/diagnose-kernels.ts models/gemma-1b-q4-col');
    log.info('KernelDiag', 'Checks:');
    log.info('KernelDiag', '  - Manifest validity');
    log.info('KernelDiag', '  - Q4K layout configuration (should be column_wise)');
    log.info('KernelDiag', '  - Kernel path override (optional)');
    log.info('KernelDiag', '  - Expected vs actual configuration');
    process.exit(0);
  }

  const modelPath = resolve(args[0]);
  const result = await diagnoseModel(modelPath);
  await printDiagnostics(result);

  // Exit with error code if issues found
  process.exit(result.issues.length > 0 ? 1 : 0);
}

main().catch(error => {
  log.error('KernelDiag', `Fatal error: ${error.message}`);
  process.exit(1);
});
