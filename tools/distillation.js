#!/usr/bin/env node

import { spawn } from 'node:child_process';

const child = spawn(
  process.execPath,
  ['tools/doppler-cli.js', 'distill', ...process.argv.slice(2)],
  {
    stdio: 'inherit',
    cwd: process.cwd(),
    env: process.env,
  }
);

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exitCode = code ?? 1;
});
