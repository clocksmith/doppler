import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { closeSync, mkdtempSync, openSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { buildRequest } from '../../src/cli/doppler-cli.js';
import { listRuntimeProfiles } from '../../src/cli/runtime-profiles.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(HERE, '..', '..');
const CLI_PATH = path.join(ROOT_DIR, 'src', 'cli', 'doppler-cli.js');

function runCli(args) {
  const logDir = mkdtempSync(path.join(tmpdir(), 'doppler-cli-runtime-profiles-'));
  const stdoutPath = path.join(logDir, 'stdout.log');
  const stderrPath = path.join(logDir, 'stderr.log');
  const stdoutFd = openSync(stdoutPath, 'w');
  const stderrFd = openSync(stderrPath, 'w');
  const result = spawnSync(process.execPath, [CLI_PATH, ...args], {
    cwd: ROOT_DIR,
    stdio: ['ignore', stdoutFd, stderrFd],
  });
  closeSync(stdoutFd);
  closeSync(stderrFd);
  const output = {
    code: result.status ?? 1,
    stdout: readFileSync(stdoutPath, 'utf8'),
    stderr: readFileSync(stderrPath, 'utf8'),
  };
  rmSync(logDir, { recursive: true, force: true });
  return output;
}

{
  const result = await listRuntimeProfiles();
  assert.equal(result.ok, true);
  assert.equal(result.schemaVersion, 1);
  const verboseTrace = result.profiles.find((profile) => profile.id === 'profiles/verbose-trace');
  assert.ok(verboseTrace, 'profiles/verbose-trace should be discoverable');
  assert.equal(verboseTrace.intent, 'investigate');
  assert.equal(verboseTrace.signals.trace, true);
  assert.equal(verboseTrace.signals.debugTokens, true);
}

{
  const result = runCli(['profiles', '--json']);
  assert.equal(result.code, 0);
  const payload = JSON.parse(result.stdout);
  assert.equal(payload.ok, true);
  assert.ok(payload.profiles.some((profile) => profile.id === 'profiles/production'));
}

{
  const result = runCli(['profiles', '--pretty']);
  assert.equal(result.code, 0);
  assert.match(result.stdout, /Runtime profiles/);
  assert.match(result.stdout, /profiles\/verbose-trace/);
}

{
  const result = await buildRequest({
    command: 'verify',
    flags: {
      config: JSON.stringify({
        request: {
          workload: 'inference',
          modelId: 'toy-model',
        },
      }),
      'runtime-profile': 'verbose-trace',
    },
  });
  assert.equal(result.request.runtimeProfile, 'profiles/verbose-trace');
}

{
  await assert.rejects(
    () => buildRequest({
      command: 'verify',
      flags: {
        config: JSON.stringify({
          request: {
            workload: 'inference',
            modelId: 'toy-model',
          },
        }),
        'runtime-profile': 'profiles/verbose-trace',
        'runtime-config': '{}',
      },
    }),
    /--runtime-profile cannot be combined with --runtime-config/
  );
}

{
  await assert.rejects(
    () => buildRequest({
      command: 'verify',
      flags: {
        config: JSON.stringify({
          request: {
            workload: 'inference',
            modelId: 'toy-model',
            runtimeProfile: 'profiles/production',
          },
        }),
        'runtime-profile': 'profiles/verbose-trace',
      },
    }),
    /--runtime-profile cannot be combined with runtimeProfile\/runtimeConfigUrl\/runtimeConfig values inside --config/
  );
}
