#!/usr/bin/env node

import { spawn } from 'node:child_process';

const LANE_MAP = Object.freeze({
  runtime_contract: Object.freeze([
    {
      label: 'verify diffusion runtime GPU-only contract',
      cmd: 'node',
      args: [
        'tools/run-node-tests.js',
        'tests/integration/diffusion-runtime-contract.test.js',
      ],
    },
  ]),
  smoke_verify_and_bench: Object.freeze([
    {
      label: 'verify deterministic diffusion verify/bench command smoke',
      cmd: 'node',
      args: [
        'tools/run-node-tests.js',
        '--force-exit',
        'tests/integration/diffusion-command-smoke.test.js',
      ],
    },
  ]),
  image_regression: Object.freeze([
    {
      label: 'verify SD3 image regression fixtures and tolerances',
      cmd: 'node',
      args: [
        'tools/run-node-tests.js',
        'tests/integration/diffusion-image-regression.test.js',
      ],
    },
  ]),
  docs_and_contract_lock: Object.freeze([
    {
      label: 'verify diffusion docs/contracts are GPU-only and stale scaffold references are removed',
      cmd: 'node',
      args: [
        'tools/run-node-tests.js',
        'tests/config/diffusion-gpu-contract-docs.test.js',
      ],
    },
  ]),
});

function parseArgs(argv) {
  const requestedLanes = [];
  let listOnly = false;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = String(argv[i] || '');
    if (arg === '--list') {
      listOnly = true;
      continue;
    }
    if (arg === '--lane') {
      const value = String(argv[i + 1] || '').trim();
      if (!value) throw new Error('--lane requires a value');
      requestedLanes.push(value);
      i += 1;
      continue;
    }
    if (arg.startsWith('--lane=')) {
      const value = arg.split('=', 2)[1].trim();
      if (!value) throw new Error('--lane requires a non-empty value');
      requestedLanes.push(value);
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  return { requestedLanes, listOnly };
}

function resolveLanes(requestedLanes) {
  if (requestedLanes.length === 0) {
    return Object.keys(LANE_MAP);
  }
  if (process.env.DOPPLER_ALLOW_OPTIONAL_DIFFUSION_LANES !== '1') {
    throw new Error(
      'Optional lane selection is disabled by default. Run full gates or set DOPPLER_ALLOW_OPTIONAL_DIFFUSION_LANES=1 for ad-hoc local lane filtering.'
    );
  }
  return requestedLanes.map((lane) => {
    if (!Object.hasOwn(LANE_MAP, lane)) {
      throw new Error(
        `Unknown lane "${lane}". Valid lanes: ${Object.keys(LANE_MAP).join(', ')}.`
      );
    }
    return lane;
  });
}

function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: 'inherit',
      env: process.env,
    });
    child.on('error', reject);
    child.on('exit', (code, signal) => {
      if (signal) {
        reject(new Error(`command terminated by signal ${signal}: ${command} ${args.join(' ')}`));
        return;
      }
      if (code !== 0) {
        reject(new Error(`command exited with code ${code}: ${command} ${args.join(' ')}`));
        return;
      }
      resolve();
    });
  });
}

async function runLane(laneName) {
  const steps = LANE_MAP[laneName];
  console.error(`[diffusion-contract-gate] lane=${laneName} start`);
  for (const step of steps) {
    console.error(`[diffusion-contract-gate] lane=${laneName} step=${step.label}`);
    // eslint-disable-next-line no-await-in-loop
    await runCommand(step.cmd, step.args);
  }
  console.error(`[diffusion-contract-gate] lane=${laneName} ok`);
}

async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  const lanes = resolveLanes(parsed.requestedLanes);
  if (parsed.listOnly) {
    for (const lane of Object.keys(LANE_MAP)) {
      console.log(lane);
    }
    return;
  }

  for (const lane of lanes) {
    // eslint-disable-next-line no-await-in-loop
    await runLane(lane);
  }
  console.error(`[diffusion-contract-gate] all lanes passed (${lanes.join(', ')})`);
}

await main();
