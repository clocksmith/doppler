/**
 * Test helpers index
 *
 * @module tests/helpers
 */

export { DemoPage } from './demo-page.js';
export { ConsoleCapture } from './console-capture.js';
export type {
  DemoPageOptions,
  GotoOptions,
  WaitForModelLoadOptions,
  WaitForGenerationOptions,
  WaitForConversionOptions,
  Stats,
} from './demo-page.js';
export type {
  LogEntry,
  AttachOptions,
  TokenQualityOptions,
  TokenQualityResult,
  LogitsInfo,
  GenerationOutput,
} from './console-capture.js';
export type {
  ModelConfig,
  ModelConfigs,
  TokenAnalysis,
  AnalyzeTokensOptions,
  TestResults,
  ParsedArgs,
} from './test-config.js';
export {
  URLS,
  MODELS,
  DEFAULT_PROMPT,
  GOOD_TOKENS,
  BAD_TOKENS,
  BASE_LOG_PATTERNS,
  analyzeTokens,
  getLogPatterns,
  isImportantLog,
  formatResults,
  parseArgs,
  printHelp,
} from './test-config.js';
