// Replays an existing browser comparison's exact prompt and output budget through
// a private llama-server. Native GGUF vs browser RDRR is a product-format
// comparison; this does not establish equal weights or an engine-only win.
import fs from 'node:fs/promises';
import { createReadStream, createWriteStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawn, execFileSync } from 'node:child_process';
import { once } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';
import os from 'node:os';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, item, i, all) => {
  if (item.startsWith('--')) pairs.push([item.slice(2), all[i + 1]]);
  return pairs;
}, []));
for (const key of ['model', 'server', 'port', 'out']) {
  if (!args[key]) throw new Error(`Missing --${key}`);
}
if (!args.comparison && !args.contract) throw new Error('Pass --comparison or --contract');
const comparison = args.comparison ? JSON.parse(await fs.readFile(args.comparison, 'utf8')) : null;
const baseline = comparison?.sections.compute?.parity ?? comparison?.sections.warm;
const contract = args.contract ? JSON.parse(await fs.readFile(args.contract, 'utf8')) : {
  promptRendered: baseline.promptContract.promptRendered,
  inputTokenIds: baseline.doppler.result.comparisonFingerprint.identity.promptTokenIds,
  maxTokens: comparison.maxTokens, warmupRuns: comparison.warmupRuns, runs: comparison.runs,
  expectedTokenIds: baseline.doppler.result.metrics.referenceTranscript?.tokens?.ids
    ?? baseline.transformersjs.generatedTokenIds,
};
const prompt = contract.promptRendered;
const sourceHash = createHash('sha256');
for await (const chunk of createReadStream(args.model)) sourceHash.update(chunk);
const report = {
  schema: 'doppler.native-request-comparison/v1',
  startedAt: new Date().toISOString(),
  comparison: args.comparison ?? null, contract,
  model: { path: args.model, sha256: sourceHash.digest('hex'), format: 'GGUF Q4_K_M' },
  serverVersion: execFileSync(args.server, ['--version'], { encoding: 'utf8' }).trim(),
  host: { hostname: os.hostname(), platform: os.platform(), arch: os.arch(), totalMemoryBytes: os.totalmem() },
  prompt, maxNewTokens: contract.maxTokens, warmupRuns: contract.warmupRuns,
  timedRuns: contract.runs, samples: [],
  claimBoundary: 'Different quantized artifacts and runtime surfaces; no equal-weight or engine-only claim. Complete latency includes local HTTP transport. First-token latency is observed at the first streamed token.',
};
const serverArgs = ['--model', args.model, '--host', '127.0.0.1', '--port', args.port,
  '--ctx-size', String(contract.inputTokenIds.length + contract.maxTokens + 1),
  '--parallel', '1', '--n-gpu-layers', '99', '--flash-attn', 'on',
  '--cache-type-k', 'f16', '--cache-type-v', 'f16'];
report.serverArgs = serverArgs;
const log = createWriteStream(`${args.out}.server.log`);
const started = performance.now();
const child = spawn(args.server, serverArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
child.stdout.pipe(log, { end: false }); child.stderr.pipe(log, { end: false });
const base = `http://127.0.0.1:${args.port}`;
try {
  for (;;) {
    if (child.exitCode !== null) throw new Error(`llama-server exited ${child.exitCode}`);
    try { if ((await fetch(`${base}/health`)).ok) break; } catch {}
    if (performance.now() - started > 300000) throw new Error('Server readiness timeout');
    await delay(250);
  }
  report.processStartToReadyMs = performance.now() - started;
  const tokenized = await (await fetch(`${base}/tokenize`, { method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ content: prompt, add_special: true, parse_special: true }),
  })).json();
  report.inputTokenIds = tokenized.tokens;
  if (JSON.stringify(tokenized.tokens) !== JSON.stringify(contract.inputTokenIds)) throw new Error('Prompt token identity mismatch');
  for (let i = 0; i < contract.warmupRuns + contract.runs; i++) {
    const begin = performance.now();
    const response = await fetch(`${base}/completion`, { method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ prompt: tokenized.tokens, n_predict: contract.maxTokens,
        temperature: 0, top_k: 1, top_p: 1, min_p: 0, repeat_penalty: 1,
        seed: 0, cache_prompt: false, return_tokens: true, stream: true }),
      signal: AbortSignal.timeout(600000),
    });
    if (!response.ok) throw new Error(`Completion HTTP ${response.status}: ${await response.text()}`);
    const decoder = new TextDecoder(); let pending = '', content = '', tokens = [], terminal = null, firstTokenMs = null;
    for await (const chunk of response.body) {
      pending += decoder.decode(chunk, { stream: true });
      const lines = pending.split('\n'); pending = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const text = line.slice(6); if (text === '[DONE]') continue;
        const event = JSON.parse(text);
        if (event.error) throw new Error(JSON.stringify(event.error));
        if (firstTokenMs === null && (event.tokens?.length || event.content)) firstTokenMs = performance.now() - begin;
        content += event.content ?? ''; tokens.push(...(event.tokens ?? []));
        if (event.stop) terminal = event;
      }
    }
    if (!terminal) throw new Error('Missing terminal completion');
    const sample = { warmup: i < contract.warmupRuns, firstTokenMs,
      completeRequestMs: performance.now() - begin, content, tokens,
      timings: terminal.timings, stopType: terminal.stop_type, truncated: terminal.truncated,
      tokensCached: terminal.tokens_cached, tokensEvaluated: terminal.tokens_evaluated,
      exactTokenMatch: contract.expectedTokenIds
        ? JSON.stringify(tokens) === JSON.stringify(contract.expectedTokenIds) : null,
      completedRequestedTokens: tokens.length === contract.maxTokens,
    };
    report.samples.push(sample);
    if (terminal.truncated || !sample.completedRequestedTokens) throw new Error('Generation did not perform the requested token work');
    console.log(JSON.stringify({ sample: i, warmup: sample.warmup, completeRequestMs: sample.completeRequestMs,
      firstTokenMs, outputTokens: tokens.length, exactTokenMatch: sample.exactTokenMatch, timings: sample.timings }));
  }
} catch (error) {
  report.error = String(error.stack ?? error); process.exitCode = 1;
} finally {
  child.kill('SIGTERM');
  await Promise.race([once(child, 'exit'), delay(5000)]);
  if (child.exitCode === null) child.kill('SIGKILL');
  log.end();
  await fs.writeFile(args.out, `${JSON.stringify(report, null, 2)}\n`);
}
