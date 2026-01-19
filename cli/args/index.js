


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

export function hasCliFlag(opts, flags) {
  return flags.some((flag) => opts.cliFlags.has(flag));
}

export function appendRuntimeConfigParams(params, opts) {
  if (opts.runtimeConfig) {
    params.set('runtimeConfig', JSON.stringify(opts.runtimeConfig));
  }
  if (opts.configChain) {
    params.set('configChain', JSON.stringify(opts.configChain));
  }
}

export function setHarnessConfig(opts, updates) {
  const harness = opts.runtimeConfig?.shared?.harness;
  if (!harness) {
    throw new Error('runtime.shared.harness is required for harness config.');
  }
  Object.assign(harness, updates);
}
