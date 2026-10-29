import fs from 'node:fs/promises';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';

const execute = promisify(execFile);
const root = path.resolve(process.argv[2]);
const plan = JSON.parse(await fs.readFile(path.join(root, process.argv[3])));
const output = path.join(root, plan.output);
await fs.mkdir(output);
const receipt = { schema: 'doppler.tuning-screen.v1', plan, attempts: [] };
const save = () => fs.writeFile(path.join(output, 'receipt.json'), JSON.stringify(receipt, null, 2));
await save();
for (const name of plan.cases) {
  const configPath = path.join(root, `${name}.json`);
  const bytes = await fs.readFile(configPath);
  const attempt = { name, configSha256: createHash('sha256').update(bytes).digest('hex'), startedAt: new Date().toISOString() };
  receipt.attempts.push(attempt);
  await save();
  console.log('START', name);
  let stdout = '', stderr = '';
  try {
    ({ stdout, stderr } = await execute(process.execPath, [path.join(root, plan.runtime ?? 'runtime', 'src/cli/doppler-cli.js'), 'bench', '--config', configPath, '--json'], {
      cwd: path.join(root, plan.runtime ?? 'runtime'), timeout: 240000, maxBuffer: 32 * 1024 * 1024,
    }));
    const raw = JSON.parse(stdout);
    if (!raw.ok) throw new Error(JSON.stringify(raw.error));
    const metrics = raw.result.metrics;
    attempt.metrics = Object.fromEntries(['totalRunMs', 'firstTokenMs', 'decodeTokensPerSec', 'decodeRecordMs', 'decodeSubmitWaitMs'].map(key => [key, metrics[key]]));
    attempt.promptIds = metrics.referenceTranscript.prompt.ids;
    attempt.outputIds = metrics.referenceTranscript.tokens.ids;
    attempt.ok = true;
    console.log('DONE', name, JSON.stringify(attempt.metrics));
  } catch (error) {
    stdout = error.stdout ?? stdout;
    stderr = error.stderr ?? stderr;
    attempt.ok = false;
    attempt.error = String(error.stack ?? error);
    console.log('FAIL', name, attempt.error);
  } finally {
    attempt.finishedAt = new Date().toISOString();
    await fs.writeFile(path.join(output, `${name}.json`), stdout);
    await fs.writeFile(path.join(output, `${name}.stderr`), stderr);
    await save();
  }
}
