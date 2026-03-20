#!/usr/bin/env node

import { spawn } from 'node:child_process';

const LANE_MAP = Object.freeze({
  command_surface_contract: Object.freeze([
    {
      label: 'verify command API and command runner parity',
      cmd: 'node',
      args: [
        'tools/run-node-tests.js',
        'tests/integration/command-api.test.js',
        'tests/integration/command-runner-shared.test.js',
        'tests/integration/training-command-malformed-surface.test.js',
        'tests/integration/ul-command-threading-parity.test.js',
        'tests/integration/distill-command-threading-parity.test.js',
      ],
    },
  ]),
  suite_routing_fail_closed: Object.freeze([
    {
      label: 'verify cross-surface suite list parity',
      cmd: 'node',
      args: [
        'tools/run-node-tests.js',
        'tests/integration/browser-harness-dispatch-map.test.js',
      ],
    },
    {
      label: 'verify unknown suites fail closed',
      cmd: 'node',
      args: [
        'tools/run-node-tests.js',
        'tests/integration/browser-harness-suite-routing.test.js',
        'tests/integration/cli-surface-contract.test.js',
      ],
    },
  ]),
  schema_validation: Object.freeze([
    {
      label: 'verify training metrics schema constraints',
      cmd: 'node',
      args: [
        'tools/run-node-tests.js',
        'tests/config/training-metrics-schema.test.js',
        'tests/config/ul-training-schema.test.js',
      ],
    },
    {
      label: 'verify checkpoint force-resume lineage schema',
      cmd: 'node',
      args: [
        'tools/run-node-tests.js',
        'tests/config/checkpoint-node-store.test.js',
        'tests/config/verify-training-provenance.test.js',
      ],
    },
    {
      label: 'verify provenance self-test policy',
      cmd: 'node',
      args: [
        'tools/verify-training-provenance.js',
        '--self-test',
      ],
    },
  ]),
  training_smoke: Object.freeze([
    {
      label: 'verify deterministic training verify/calibrate smoke',
      cmd: 'node',
      args: [
        'tools/run-node-tests.js',
        '--force-exit',
        'tests/integration/training-intent-split.test.js',
        'tests/integration/training-report-lineage.test.js',
        'tests/integration/training-force-resume-lineage.test.js',
      ],
    },
  ]),
  rollout_governance: Object.freeze([
    {
      label: 'verify distill studio rollout contracts',
      cmd: 'node',
      args: [
        'tools/run-node-tests.js',
        'tests/config/distill-studio-diagnostics.test.js',
        'tests/config/distill-studio-mvp.test.js',
        'tests/config/distill-studio-quality-gate.test.js',
        'tests/config/training-workload-packs.test.js',
        'tests/config/training-report-id-publication.test.js',
      ],
    },
    {
      label: 'verify deterministic training workload packs + report-id bindings',
      cmd: 'node',
      args: [
        'tools/verify-training-workload-packs.js',
        '--registry',
        'src/training/workload-packs/registry.json',
      ],
    },
    {
      label: 'emit machine-readable training report-id traceability artifact',
      cmd: 'node',
      args: [
        'tools/publish-training-report-ids.js',
        '--registry',
        'src/training/workload-packs/registry.json',
        '--out',
        '/tmp/doppler-training-report-ids.json',
      ],
    },
    {
      label: 'emit machine-readable training contract delta artifact',
      cmd: 'node',
      args: [
        'tools/emit-training-contract-delta.js',
        '--out',
        '/tmp/doppler-training-contract-delta.json',
        '--note',
        'ci-gate',
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
      if (!value) {
        throw new Error('--lane requires a value');
      }
      requestedLanes.push(value);
      i += 1;
      continue;
    }
    if (arg.startsWith('--lane=')) {
      const value = arg.split('=', 2)[1].trim();
      if (!value) {
        throw new Error('--lane requires a non-empty value');
      }
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
  if (process.env.DOPPLER_ALLOW_OPTIONAL_TRAINING_LANES !== '1') {
    throw new Error(
      'Optional lane selection is disabled by default. Run full gates or set DOPPLER_ALLOW_OPTIONAL_TRAINING_LANES=1 for ad-hoc local lane filtering.'
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
  console.error(`[training-contract-gate] lane=${laneName} start`);
  for (const step of steps) {
    console.error(`[training-contract-gate] lane=${laneName} step=${step.label}`);
    // eslint-disable-next-line no-await-in-loop
    await runCommand(step.cmd, step.args);
  }
  console.error(`[training-contract-gate] lane=${laneName} ok`);
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
  console.error(`[training-contract-gate] all lanes passed (${lanes.join(', ')})`);
}

await main();
