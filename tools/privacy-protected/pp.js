#!/usr/bin/env node
// Protected-route command line (JavaScript only; it replaces the removed Go `pp` binary with the same subcommands):
//
//   node tools/privacy-protected/pp.js <command> [flags]
//
//   requester    the key holder's session driver (whole model, dual check, homomorphic split, trap lanes)
//   bridge       the requester's key holder for a host process, over stdin/stdout
//   gpu-compile  compile a session program for the WebGPU evaluator (webgpu/evaluator.js)
//   compare      check a requester report against the plaintext references (after the route closed)
//   reference    the pinned continuation of a prompt
//   audit        the approximated circuit against the float64 model over every three-letter prompt
//   plain-bench  time the plaintext forward pass
//   trapset      a signed public known-answer trap set (v1)
//   trap-oracle  the secret fresh trap oracle (v2)
//
// The route stays off by default everywhere: nothing here runs unless invoked, and the insecure development profile
// is refused without --allow-insecure-dev.

const COMMANDS = {
  requester: async (a) => (await import('./requester.js')).main(a),
  bridge: async (a) => (await import('./bridge.js')).main(a),
  'gpu-compile': async (a) => (await import('./tools.js')).gpuCompile(a),
  compare: async (a) => (await import('./tools.js')).compare(a),
  reference: async (a) => (await import('./tools.js')).reference(a),
  audit: async (a) => (await import('./tools.js')).audit(a),
  'plain-bench': async (a) => (await import('./tools.js')).plainBench(a),
  trapset: async (a) => (await import('./trapset.js')).main(a),
  'trap-oracle': async (a) => (await import('./trap-oracle.js')).main(a),
};

const [cmd, ...rest] = process.argv.slice(2);
if (!COMMANDS[cmd]) {
  console.error(`usage: pp.js ${Object.keys(COMMANDS).join('|')} [flags]`);
  process.exit(2);
}
try {
  await COMMANDS[cmd](rest);
} catch (e) {
  console.error(`error: ${e.message}`);
  process.exit(1);
}
