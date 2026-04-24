/**
 * Tooling Surface Exports
 *
 * Internal tooling, diagnostics, and infrastructure used by demo, CLI,
 * and harness code. Not part of the core inference API.
 *
 * @module tooling-exports
 */

// Shared browser-safe tooling exports.
export * from './tooling-exports.shared.js';

// Node-only tooling exports.
export { runNodeCommand, normalizeNodeCommand, hasNodeWebGPUSupport } from './tooling/node-command-runner.js';
export { runBrowserCommandInNode, normalizeNodeBrowserCommand } from './tooling/node-browser-command-runner.js';
export {
  exportProgramBundle,
  writeProgramBundle,
  loadProgramBundle,
  checkProgramBundleFile,
} from './tooling/program-bundle.js';
export {
  PROGRAM_BUNDLE_PARITY_SCHEMA_ID,
  checkProgramBundleParity,
} from './tooling/program-bundle-parity.js';
export {
  buildManifestIntegrityFromModelDir,
  refreshManifestIntegrity,
} from './tooling/rdrr-integrity-refresh.js';
export type {
  NodeCommandRunOptions,
  NodeCommandRunResult,
} from './tooling/node-command-runner.js';
export type {
  NodeBrowserCommandRunOptions,
} from './tooling/node-browser-command-runner.js';
export type {
  ProgramBundleCheckResult,
  ProgramBundleExportOptions,
  ProgramBundleWriteResult,
} from './tooling/program-bundle.js';
export type {
  ProgramBundleParityOptions,
  ProgramBundleParityResult,
} from './tooling/program-bundle-parity.js';
