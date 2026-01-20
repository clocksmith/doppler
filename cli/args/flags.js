



export const FLAG_SPECS = [
  { names: ['--help', '-h'], handler: (opts) => { opts.help = true; } },
  { names: ['--config'], handler: (opts, tokens) => { opts.config = tokens.shift() || null; } },
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
