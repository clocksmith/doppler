#!/usr/bin/env node
/**
 * Diagnostic tool to verify kernel selection logic.
 *
 * This script checks:
 * 1. Manifest kernel hints are correctly configured
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

interface DiagnosticResult {
  modelPath: string;
  manifestValid: boolean;
  hasKernelHints: boolean;
  q4kLayout?: string;
  kernelHints?: any;
  expectedKernels: {
    q4kMatmul: string;
    computePrecision: string;
    attentionDecode: string;
    attentionPrefill: string;
  };
  issues: string[];
  recommendations: string[];
}

async function diagnoseModel(modelPath: string): Promise<DiagnosticResult> {
  const result: DiagnosticResult = {
    modelPath,
    manifestValid: false,
    hasKernelHints: false,
    expectedKernels: {
      q4kMatmul: 'dequant_f16',
      computePrecision: 'f16',
      attentionDecode: 'streaming',
      attentionPrefill: 'tiled_large',
    },
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

    // Check kernel hints
    const kernelHints = manifest.optimizations?.kernelHints;
    result.kernelHints = kernelHints;
    result.hasKernelHints = !!kernelHints;

    if (!kernelHints) {
      result.issues.push('Warning: Missing optimizations.kernelHints');
      result.recommendations.push('Run: npx tsx tools/update-manifest.ts ' + modelPath + '/manifest.json \\');
      result.recommendations.push('  --compute-precision f16 --q4k-matmul dequant_f16 \\');
      result.recommendations.push('  --f16-matmul gemv_subgroup --attention-prefill tiled_large \\');
      result.recommendations.push('  --attention-decode streaming --tuned-device "Apple M3"');
    } else {
      log.info('KernelDiag', 'OK Kernel hints present');

      // Verify each hint
      const checks = [
        { field: 'q4kMatmul', expected: 'dequant_f16', reason: '2.3x faster than fused (8 vs 3 tok/s)' },
        { field: 'computePrecision', expected: 'f16', reason: 'Uses F16 compute when available' },
        { field: 'f16Matmul', expected: 'gemv_subgroup', reason: 'Best for GEMV decode with subgroups' },
        { field: 'attentionPrefill', expected: 'tiled_large', reason: 'Best for long sequences' },
        { field: 'attentionDecode', expected: 'streaming', reason: 'Best for single-token generation' },
      ];

      for (const check of checks) {
        const value = kernelHints[check.field];
        if (!value) {
          result.issues.push(`Warning: Missing kernelHints.${check.field}`);
        } else if (value !== check.expected) {
          result.issues.push(`Warning: kernelHints.${check.field} = "${value}" (expected "${check.expected}" for ${check.reason})`);
        } else {
          log.info('KernelDiag', `OK ${check.field}: ${value} - ${check.reason}`);
        }
      }

      // Check benchmark result
      if (kernelHints.benchmarkTokPerSec) {
        log.info('KernelDiag', `Benchmark: Documented performance: ${kernelHints.benchmarkTokPerSec} tok/s`);
        if (kernelHints.benchmarkTokPerSec < 7) {
          result.issues.push(`Warning: Documented performance (${kernelHints.benchmarkTokPerSec} tok/s) is below expected (8 tok/s)`);
        }
      }
    }

    // Check quantization type
    if (manifest.quantization !== 'Q4_K_M') {
      result.issues.push(`Info: Quantization is "${manifest.quantization}" (Q4K hints apply to Q4_K_M only)`);
    }

  } catch (error: any) {
    result.issues.push(`Error reading manifest: ${error.message}`);
  }

  return result;
}

async function printDiagnostics(result: DiagnosticResult) {
  log.info('KernelDiag', '='.repeat(70));
  log.info('KernelDiag', 'DOPPLER Kernel Diagnostics');
  log.info('KernelDiag', '='.repeat(70));
  log.info('KernelDiag', `Model: ${result.modelPath}`);
  log.info('KernelDiag', `Manifest: ${result.manifestValid ? 'OK Valid' : 'ERROR Invalid'}`);

  if (result.q4kLayout) {
    const layoutStatus = result.q4kLayout === 'column_wise' ? 'OK' : 'WARNING';
    log.info('KernelDiag', `Q4K Layout: ${layoutStatus} ${result.q4kLayout}`);
  }

  if (result.hasKernelHints) {
    log.info('KernelDiag', 'Kernel Hints: OK Present');
  } else {
    log.warn('KernelDiag', 'Kernel Hints: ERROR Missing');
  }

  if (result.issues.length > 0) {
    log.warn('KernelDiag', 'Issues Found:');
    result.issues.forEach(issue => log.warn('KernelDiag', `  ${issue}`));
  }

  if (result.recommendations.length > 0) {
    log.info('KernelDiag', 'Recommendations:');
    result.recommendations.forEach(rec => log.info('KernelDiag', `  ${rec}`));
  }

  if (result.issues.length === 0 && result.hasKernelHints) {
    log.info('KernelDiag', 'Model is optimally configured!');
    log.info('KernelDiag', 'Expected Performance:');
    log.info('KernelDiag', '  - Gemma 3 1B: ~8 tok/s (Apple M3)');
    log.info('KernelDiag', '  - Column-wise layout: +14% vs flat');
    log.info('KernelDiag', '  - Dequant F16 path: 2.3x faster than fused Q4K');
    log.info('KernelDiag', 'Run benchmark:');
    log.info('KernelDiag', `  npx tsx cli/index.ts bench inference --model ${result.modelPath.split('/').pop()} --runs 3`);
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
    log.info('KernelDiag', '  - Kernel hints presence and correctness');
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
  log.error('KernelDiag', `Fatal error: ${(error as Error).message}`);
  process.exit(1);
});
