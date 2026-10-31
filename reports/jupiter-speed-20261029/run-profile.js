import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
const root = path.dirname(fileURLToPath(import.meta.url));
const execute = promisify(execFile);
const contract = JSON.parse(await fs.readFile(path.join(root, 'contract.json')));
const source = path.join(root, 'runtime');
const prior = '$HOME/deco/hatch/pooled-doppler-components-20261029/deep-profile/receipts/0-manifest-default-config.json';
const baseline = JSON.parse(await fs.readFile(prior));
const out = path.join(root, 'runs');
await fs.mkdir(out); // Refuse overwriting an earlier campaign.
const receipt = { schema: 'doppler.amd-host-profile-validation.v1', contractSha256: createHash('sha256').update(await fs.readFile(path.join(root, 'contract.json'))).digest('hex'), sessions: [], failures: [] };
const save = () => fs.writeFile(path.join(out, 'receipt.json'), JSON.stringify(receipt, null, 2));
await save();
for (const prompt of contract.prompts) {
  const text = (prompt.prefix ?? '').repeat(prompt.prefixRepeats ?? 0) + prompt.text;
  const formatted = '<|im_start|>user\n' + text + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';
  for (let pair = 0; pair < contract.pairsPerPrompt; pair++) {
    for (const lane of pair % 2 ? ['candidate', 'baseline'] : ['baseline', 'candidate']) {
      const id = `${prompt.id}-${pair}-${lane}`;
      const config = structuredClone(baseline);
      config.request.runtimeConfig.shared.benchmark.run.customPrompt = formatted;
      config.request.runtimeConfig.inference.prompt = formatted;
      config.request.workload = 'inference';
      if (lane === 'candidate') config.request.runtimeProfile = contract.candidate;
      config.run.browser.userDataDir = path.join(root, 'browser-' + lane);
      config.run.browser.port = 19912;
      config.run.bench.save = false;
      const configPath = path.join(out, id + '-config.json');
      await fs.writeFile(configPath, JSON.stringify(config, null, 2));
      console.log('START', id);
      let stdout = '', stderr = '';
      const start = Date.now();
      try {
        ({ stdout, stderr } = await execute(process.execPath, [path.join(source, 'src/cli/doppler-cli.js'), 'bench', '--config', configPath, '--json'], { cwd: source, timeout: 240000, maxBuffer: 32 * 1024 * 1024 }));
        const raw = JSON.parse(stdout);
        if (!raw.ok) throw new Error(JSON.stringify(raw.error));
        const result = raw.result, metrics = result.metrics;
        receipt.sessions.push({ id, prompt: prompt.id, pair, lane, elapsedMs: Date.now() - start, outputIds: metrics.referenceTranscript.tokens.ids, promptIds: metrics.referenceTranscript.prompt.ids, metrics, timing: result.timing, device: result.deviceInfo, memoryStats: result.memoryStats, request: result.request });
        console.log('DONE', id, metrics.decodeTokensPerSec, 'tok/s', metrics.totalRunMs, 'ms request');
      } catch (error) {
        stdout = error.stdout ?? stdout;
        stderr = error.stderr ?? stderr;
        receipt.failures.push({ id, error: String(error.stack ?? error) });
      } finally {
        await fs.writeFile(path.join(out, id + '.json'), stdout);
        await fs.writeFile(path.join(out, id + '.stderr'), stderr);
        await save();
      }
    }
  }
}
receipt.complete = true;
await save();
