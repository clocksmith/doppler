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


export function parseArgs(argv) {
  const opts = {
    command: null,
    suite: null,
    model: null,
    baseUrl: null,
    config: null,
    runtimeConfig: null,
    configChain: null,
    noServer: null,
    headless: null,
    minimized: null,
    reuseBrowser: null,
    cdpEndpoint: null,
    verbose: false,
    filter: null,
    timeout: null,
    output: null,
    html: null,
    compare: null,
    profileDir: null,
    retries: null,
    quiet: false,
    help: false,
    perf: false,
  };

  const tokens = [...argv];
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
      handler(opts, tokens);
      continue;
    }
    throw new Error(`Unexpected argument "${arg}". Use --config for CLI configuration.`);
  }

  return opts;
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
