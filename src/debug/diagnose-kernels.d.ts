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
 *   doppler --config <ref>
 *
 * Config:
 *   - model (string, required)
 *   - tools.diagnoseKernels.modelPath (string, optional; overrides models/<model>)
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
