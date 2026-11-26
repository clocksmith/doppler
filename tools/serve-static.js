#!/usr/bin/env node
// Serve the repository (or --root) over HTTP on 127.0.0.1 for local pages such as
// tests/harness.html. Runs until Ctrl-C.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createStaticFileServer } from '../src/tooling/node-browser/static-server.js';

const USAGE = 'Usage: node tools/serve-static.js [--port 8080] [--root .]';
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function parseArgs(argv) {
  const parsed = { port: 8080, root: repoRoot };
  for (let i = 0; i < argv.length; i += 2) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (flag === '--help' || flag === '-h') {
      console.log(USAGE);
      process.exit(0);
    }
    if (value == null || value.startsWith('--')) throw new Error(`Missing value for ${flag}. ${USAGE}`);
    if (flag === '--port') {
      const port = Number(value);
      if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error(`--port must be an integer from 0 to 65535, got "${value}".`);
      parsed.port = port;
    } else if (flag === '--root') {
      parsed.root = path.resolve(value);
    } else {
      throw new Error(`Unknown flag: ${flag}. ${USAGE}`);
    }
  }
  return parsed;
}

const { port, root } = parseArgs(process.argv.slice(2));
const server = await createStaticFileServer({ rootDir: root, host: '127.0.0.1', port });
console.log(`Serving ${root} at ${server.baseUrl}/ (Ctrl-C to stop)`);

async function shutdown() {
  await server.close();
  process.exit(0);
}
process.once('SIGINT', shutdown);
process.once('SIGTERM', shutdown);
