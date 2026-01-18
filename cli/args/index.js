

/**
 * CLI argument parsing.
 */

import { FLAG_HANDLERS } from './flags.js';
import { resolveFlagAlias, suggestClosestFlags } from './suggestions.js';

export { FLAG_SPECS, FLAG_HANDLERS, KNOWN_FLAGS } from './flags.js';
export {
  levenshteinDistance,
  normalizeFlag,
  suggestFlag,
  resolveFlagAlias,
  suggestClosestFlags,
} from './suggestions.js';

/**
 * @typedef {Object} CLIOptions
 * @property {Set<string>} cliFlags
 * @property {string} command
 * @property {string} suite
 * @property {string} model
 * @property {string} baseUrl
 * @property {string|null} config
 * @property {string|null} mode
 * @property {object|null} runtimeConfig
 * @property {string[]|null} configChain
 * @property {boolean} dumpConfig
 * @property {boolean} listPresets
 * @property {boolean} noServer
 * @property {boolean} headless
 * @property {boolean} minimized
 * @property {boolean} reuseBrowser
 * @property {string} cdpEndpoint
 * @property {boolean} verbose
 * @property {string|null} filter
 * @property {number} timeout
 * @property {string|null} output
 * @property {string|null} html
 * @property {string|null} compare
 * @property {string|null} profileDir
 * @property {number} retries
 * @property {boolean} quiet
 * @property {boolean} help
 * @property {boolean} perf
 * @property {boolean} skipLoad
 * @property {boolean} warm
 */

/**
 * Normalize suite name (handle legacy aliases).
 * @param {string} suite
 * @returns {string}
 */
export function normalizeSuite(suite) {
  const legacyMap = {
    'bench:kernels': 'kernels',
    'bench:pipeline': 'inference',
    'bench:system': 'system',
    'correctness': 'kernels',
    'simulate': 'simulation',
  };
  return legacyMap[suite] || suite;
}

/**
 * Parse command line arguments.
 * @param {string[]} argv
 * @returns {CLIOptions}
 */
export function parseArgs(argv) {
  const opts = {
    cliFlags: new Set(),
    command: 'test',
    suite: 'kernels',
    model: 'gemma-2-2b-it-wf16',
    baseUrl: 'http://localhost:8080',
    config: null,
    mode: null,
    runtimeConfig: null,
    configChain: null,
    dumpConfig: false,
    listPresets: false,
    noServer: false,
    headless: true,
    minimized: false,
    reuseBrowser: true,
    cdpEndpoint: 'http://localhost:9222',
    verbose: false,
    filter: null,
    timeout: 300000,
    output: null,
    html: null,
    compare: null,
    profileDir: null,
    retries: 2,
    quiet: false,
    help: false,
    perf: false,
    skipLoad: false,
    warm: false,
  };

  const tokens = [...argv];
  let positionalIndex = 0;

  while (tokens.length) {
    const arg = tokens.shift();
    if (arg.startsWith('-')) {
      let resolvedFlag = arg;
      let handler = FLAG_HANDLERS.get(arg);
      if (!handler) {
        const alias = resolveFlagAlias(arg);
        if (alias) {
          resolvedFlag = alias;
          handler = FLAG_HANDLERS.get(alias);
        }
      }
      if (!handler) {
        const suggestions = suggestClosestFlags(arg);
        const hint = suggestions.length > 0
          ? ` Did you mean ${suggestions.map((s) => `"${s}"`).join(' or ')}?`
          : '';
        throw new Error(`Unknown flag "${arg}".${hint}`);
      }
      opts.cliFlags.add(resolvedFlag);
      handler(opts, tokens);
      continue;
    }

    if (positionalIndex === 0) {
      if (arg === 'run' || arg === 'test' || arg === 'bench' || arg === 'debug') {
        opts.command = arg;
      } else {
        opts.suite = normalizeSuite(arg);
      }
    } else if (positionalIndex === 1) {
      opts.suite = normalizeSuite(arg);
    }
    positionalIndex++;
  }

  return opts;
}

/**
 * Check if CLI flags were explicitly provided.
 * @param {CLIOptions} opts
 * @param {string[]} flags
 * @returns {boolean}
 */
export function hasCliFlag(opts, flags) {
  return flags.some((flag) => opts.cliFlags.has(flag));
}

/**
 * Append runtime config to URL params.
 * @param {URLSearchParams} params
 * @param {CLIOptions} opts
 */
export function appendRuntimeConfigParams(params, opts) {
  if (opts.runtimeConfig) {
    params.set('runtimeConfig', JSON.stringify(opts.runtimeConfig));
  }
  if (opts.configChain) {
    params.set('configChain', JSON.stringify(opts.configChain));
  }
}

/**
 * Set harness-specific config in runtime config.
 * @param {CLIOptions} opts
 * @param {object} updates
 */
export function setHarnessConfig(opts, updates) {
  const harness = opts.runtimeConfig?.shared?.harness;
  if (!harness) {
    throw new Error('runtime.shared.harness is required for harness config.');
  }
  Object.assign(harness, updates);
}
