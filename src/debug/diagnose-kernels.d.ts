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

export interface DiagnosticResult {
  modelPath: string;
  manifestValid: boolean;
  hasKernelPath: boolean;
  q4kLayout?: string;
  kernelPath?: unknown;
  issues: string[];
  recommendations: string[];
}
