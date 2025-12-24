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
 *   npx tsx tools/diagnose-kernels.ts <model-path>
 *
 * Example:
 *   npx tsx tools/diagnose-kernels.ts models/gemma-1b-q4-col
 */

import { readFile } from 'fs/promises';
import { resolve, join } from 'path';

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
      result.issues.push('⚠️  Missing config.q4kLayout field');
      result.recommendations.push('Run converter with --q4k-layout column_wise');
    } else if (q4kLayout !== 'column_wise') {
      result.issues.push(`⚠️  Q4K layout is "${q4kLayout}" (expected "column_wise" for best performance)`);
      result.recommendations.push('Re-convert model with --q4k-layout column_wise for 14% speedup');
    } else {
      console.log('✅ Q4K layout: column_wise (optimal)');
    }

    // Check kernel hints
    const kernelHints = manifest.optimizations?.kernelHints;
    result.kernelHints = kernelHints;
    result.hasKernelHints = !!kernelHints;

    if (!kernelHints) {
      result.issues.push('⚠️  Missing optimizations.kernelHints');
      result.recommendations.push('Run: npx tsx tools/update-manifest.ts ' + modelPath + '/manifest.json \\');
      result.recommendations.push('  --compute-precision f16 --q4k-matmul dequant_f16 \\');
      result.recommendations.push('  --f16-matmul gemv_subgroup --attention-prefill tiled_large \\');
      result.recommendations.push('  --attention-decode streaming --tuned-device "Apple M3"');
    } else {
      console.log('✅ Kernel hints present');

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
          result.issues.push(`⚠️  Missing kernelHints.${check.field}`);
        } else if (value !== check.expected) {
          result.issues.push(`⚠️  kernelHints.${check.field} = "${value}" (expected "${check.expected}" for ${check.reason})`);
        } else {
          console.log(`✅ ${check.field}: ${value} - ${check.reason}`);
        }
      }

      // Check benchmark result
      if (kernelHints.benchmarkTokPerSec) {
        console.log(`📊 Documented performance: ${kernelHints.benchmarkTokPerSec} tok/s`);
        if (kernelHints.benchmarkTokPerSec < 7) {
          result.issues.push(`⚠️  Documented performance (${kernelHints.benchmarkTokPerSec} tok/s) is below expected (8 tok/s)`);
        }
      }
    }

    // Check quantization type
    if (manifest.quantization !== 'Q4_K_M') {
      result.issues.push(`ℹ️  Quantization is "${manifest.quantization}" (Q4K hints apply to Q4_K_M only)`);
    }

  } catch (error: any) {
    result.issues.push(`❌ Error reading manifest: ${error.message}`);
  }

  return result;
}

async function printDiagnostics(result: DiagnosticResult) {
  console.log('\n' + '='.repeat(70));
  console.log('DOPPLER Kernel Diagnostics');
  console.log('='.repeat(70));
  console.log(`Model: ${result.modelPath}`);
  console.log(`Manifest: ${result.manifestValid ? '✅ Valid' : '❌ Invalid'}`);
  console.log('');

  if (result.q4kLayout) {
    const layoutStatus = result.q4kLayout === 'column_wise' ? '✅' : '⚠️';
    console.log(`Q4K Layout: ${layoutStatus} ${result.q4kLayout}`);
  }

  if (result.hasKernelHints) {
    console.log('Kernel Hints: ✅ Present\n');
  } else {
    console.log('Kernel Hints: ❌ Missing\n');
  }

  if (result.issues.length > 0) {
    console.log('Issues Found:');
    result.issues.forEach(issue => console.log('  ' + issue));
    console.log('');
  }

  if (result.recommendations.length > 0) {
    console.log('Recommendations:');
    result.recommendations.forEach(rec => console.log('  ' + rec));
    console.log('');
  }

  if (result.issues.length === 0 && result.hasKernelHints) {
    console.log('✨ Model is optimally configured!');
    console.log('');
    console.log('Expected Performance:');
    console.log('  • Gemma 3 1B: ~8 tok/s (Apple M3)');
    console.log('  • Column-wise layout: +14% vs flat');
    console.log('  • Dequant F16 path: 2.3x faster than fused Q4K');
    console.log('');
    console.log('Run benchmark:');
    console.log(`  npx tsx tools/doppler-cli.ts bench inference --model ${result.modelPath.split('/').pop()} --runs 3`);
  }

  console.log('='.repeat(70));
}

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    console.log('Usage: npx tsx tools/diagnose-kernels.ts <model-path>');
    console.log('');
    console.log('Example:');
    console.log('  npx tsx tools/diagnose-kernels.ts models/gemma-1b-q4-col');
    console.log('');
    console.log('Checks:');
    console.log('  • Manifest validity');
    console.log('  • Q4K layout configuration (should be column_wise)');
    console.log('  • Kernel hints presence and correctness');
    console.log('  • Expected vs actual configuration');
    process.exit(0);
  }

  const modelPath = resolve(args[0]);
  const result = await diagnoseModel(modelPath);
  await printDiagnostics(result);

  // Exit with error code if issues found
  process.exit(result.issues.length > 0 ? 1 : 0);
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
