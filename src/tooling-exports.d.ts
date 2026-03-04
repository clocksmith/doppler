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
export type {
  NodeCommandRunOptions,
  NodeCommandRunResult,
} from './tooling/node-command-runner.js';
export type {
  NodeBrowserCommandRunOptions,
} from './tooling/node-browser-command-runner.js';
