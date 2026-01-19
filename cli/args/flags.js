



export const FLAG_SPECS = [
  { names: ['--help', '-h'], handler: (opts) => { opts.help = true; } },
  { names: ['--config'], handler: (opts, tokens) => { opts.config = tokens.shift() || null; } },
  { names: ['--mode'], handler: (opts, tokens) => { opts.mode = tokens.shift() || null; } },
  { names: ['--dump-config'], handler: (opts) => { opts.dumpConfig = true; } },
  { names: ['--list-presets'], handler: (opts) => { opts.listPresets = true; } },
  { names: ['--model', '-m'], handler: (opts, tokens) => { opts.model = tokens.shift() || opts.model; } },
  { names: ['--base-url', '-u'], handler: (opts, tokens) => { opts.baseUrl = tokens.shift() || opts.baseUrl; } },
  { names: ['--no-server'], handler: (opts) => { opts.noServer = true; } },
  { names: ['--headless'], handler: (opts) => { opts.headless = true; } },
  { names: ['--headed', '--no-headless'], handler: (opts) => { opts.headless = false; } },
  { names: ['--minimized', '--no-focus'], handler: (opts) => { opts.minimized = true; } },
  { names: ['--reuse-browser'], handler: (opts) => { opts.reuseBrowser = true; } },
  { names: ['--no-reuse-browser', '--new-browser'], handler: (opts) => { opts.reuseBrowser = false; } },
  { names: ['--cdp-endpoint'], handler: (opts, tokens) => { opts.cdpEndpoint = tokens.shift() || opts.cdpEndpoint; } },
  { names: ['--skip-load'], handler: (opts) => { opts.skipLoad = true; } },
  { names: ['--warm'], handler: (opts) => { opts.warm = true; opts.headless = false; opts.reuseBrowser = true; } },
  { names: ['--inference'], handler: (opts) => { opts.suite = 'inference'; } },
  { names: ['--simulation', '--simulate'], handler: (opts) => { opts.suite = 'simulation'; } },
  { names: ['--kernels'], handler: (opts) => { opts.suite = 'kernels'; } },
  { names: ['--training'], handler: (opts) => { opts.suite = 'training'; } },
  { names: ['--quick'], handler: (opts) => { opts.suite = 'quick'; } },
  { names: ['--full'], handler: (opts) => { opts.suite = 'all'; } },
  { names: ['--filter', '-f'], handler: (opts, tokens) => { opts.filter = tokens.shift() || null; } },
  { names: ['--timeout'], handler: (opts, tokens) => { opts.timeout = parseInt(tokens.shift() || '120000', 10); } },
  { names: ['--output', '-o'], handler: (opts, tokens) => { opts.output = tokens.shift() || null; } },
  { names: ['--html'], handler: (opts, tokens) => { opts.html = tokens.shift() || null; } },
  { names: ['--compare', '-c'], handler: (opts, tokens) => { opts.compare = tokens.shift() || null; } },
  { names: ['--profile-dir'], handler: (opts, tokens) => { opts.profileDir = tokens.shift() || null; } },
  { names: ['--retries'], handler: (opts, tokens) => { opts.retries = parseInt(tokens.shift() || '2', 10); } },
  { names: ['--perf'], handler: (opts) => { opts.perf = true; } },
];

function buildFlagHandlers() {
  const handlers = new Map();
  for (const spec of FLAG_SPECS) {
    for (const name of spec.names) {
      handlers.set(name, spec.handler);
    }
  }
  return handlers;
}

export const FLAG_HANDLERS = buildFlagHandlers();

export const KNOWN_FLAGS = new Set(FLAG_HANDLERS.keys());
