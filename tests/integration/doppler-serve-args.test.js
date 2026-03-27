import assert from 'node:assert/strict';
import { parseServeArgs } from '../../src/cli/doppler-serve.js';

// Default values
{
  const args = parseServeArgs([]);
  assert.equal(args.port, 8080);
  assert.equal(args.host, '127.0.0.1');
  assert.equal(args.model, null);
  assert.equal(args.help, false);
}

// Custom port
{
  const args = parseServeArgs(['--port', '3000']);
  assert.equal(args.port, 3000);
}

// Custom host
{
  const args = parseServeArgs(['--host', '0.0.0.0']);
  assert.equal(args.host, '0.0.0.0');
}

// Model flag
{
  const args = parseServeArgs(['--model', 'gemma3-270m']);
  assert.equal(args.model, 'gemma3-270m');
}

// Help flag
{
  const args = parseServeArgs(['--help']);
  assert.equal(args.help, true);
}

{
  const args = parseServeArgs(['-h']);
  assert.equal(args.help, true);
}

// Combined flags
{
  const args = parseServeArgs(['--model', 'qwen3-0.8b', '--port', '9090', '--host', '0.0.0.0']);
  assert.equal(args.model, 'qwen3-0.8b');
  assert.equal(args.port, 9090);
  assert.equal(args.host, '0.0.0.0');
}

// Invalid port
assert.throws(
  () => parseServeArgs(['--port', 'abc']),
  /valid port number/
);

assert.throws(
  () => parseServeArgs(['--port', '-1']),
  /valid port number/
);

assert.throws(
  () => parseServeArgs(['--port', '99999']),
  /valid port number/
);

// Missing value
assert.throws(
  () => parseServeArgs(['--port']),
  /Missing value/
);

assert.throws(
  () => parseServeArgs(['--model']),
  /Missing value/
);

// Unknown flag
assert.throws(
  () => parseServeArgs(['--unknown']),
  /Unknown flag/
);

// Positional arg rejected
assert.throws(
  () => parseServeArgs(['something']),
  /Unexpected positional/
);

console.log('doppler-serve-args.test: ok');
