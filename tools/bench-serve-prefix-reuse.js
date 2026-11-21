#!/usr/bin/env node
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const [modelId, modelUrl, outputPath] = process.argv.slice(2);
if (!modelId || !modelUrl || !outputPath) {
  console.error('Usage: node tools/bench-serve-prefix-reuse.js <modelId> <modelUrl> <output.json>');
  process.exit(2);
}

const SHARED_SYSTEM = [
  'You are a deterministic technical systems analyst evaluating WebGPU memory hierarchy, hybrid attention KV caches, and Gated DeltaNet recurrent state checkpoints.',
  'Always answer concisely in one short factual sentence using greedy decoding.',
  'Reference architecture facts: Layer pattern alternates three linear-attention Gated DeltaNet blocks with one full-attention KV block across 24 layers.',
  'Prefix caching retains both the full-attention K/V buffers and the recurrent linear-attention state snapshot at the assistant turn boundary so follow-up turns prefill only the newly appended user delta tokens.',
  'Memory accounting charges exact allocated GPU buffer bytes per retained GenerationContext entry inside a bounded LRU byte pool on loopback hosts.',
  'When edited history diverges from the cached token prefix, prefix matching truncates or misses cleanly with zero stale token leakage across conversations.',
].join(' ');

const TURNS = [
  'State the two kinds of per-layer state retained at a turn boundary.',
  'Now state what follow-up turns prefill when the prefix matches.',
  'Finally state what happens when history diverges from the cached prefix.',
];

async function startServer(port, enableCache) {
  const args = [
    path.join(REPO_ROOT, 'src/cli/doppler-serve.js'),
    '--host', '127.0.0.1',
    '--port', String(port),
    '--model', modelId,
    '--model-url', modelUrl,
    enableCache ? '--prefix-cache' : '--no-prefix-cache',
  ];
  const proc = spawn(process.execPath, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  proc.stderr.on('data', (chunk) => { stderr += chunk.toString('utf8'); });

  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    if (proc.exitCode !== null) {
      throw new Error(`doppler-serve exited early (${proc.exitCode}): ${stderr}`);
    }
    try {
      const res = await fetch(`http://127.0.0.1:${port}/health`);
      if (res.ok) {
        const body = await res.json();
        if (body.status === 'ok' || body.ok === true) return { proc, initialHealth: body };
      }
    } catch {
      // Wait for listen.
    }
    await new Promise((r) => setTimeout(r, 150));
  }
  proc.kill('SIGTERM');
  throw new Error(`Timed out waiting for doppler-serve on port ${port}: ${stderr}`);
}

async function runConversation(port) {
  const messages = [{ role: 'system', content: SHARED_SYSTEM }];
  const turnMetrics = [];
  for (let i = 0; i < TURNS.length; i += 1) {
    messages.push({ role: 'user', content: TURNS[i] });
    const t0 = performance.now();
    const res = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: modelId,
        messages,
        max_tokens: 16,
        temperature: 0,
        top_k: 1,
        include_receipt: true,
      }),
    });
    const wallMs = performance.now() - t0;
    assert.equal(res.status, 200, `Expected HTTP 200 on turn ${i + 1}, got ${res.status}`);
    const payload = await res.json();
    const content = payload.choices[0].message.content;
    messages.push({ role: 'assistant', content });
    const stats = payload.doppler_receipt?.timing ?? payload.doppler_receipt?.evidence?.stats ?? {};
    const health = await (await fetch(`http://127.0.0.1:${port}/health`)).json();
    turnMetrics.push({
      turn: i + 1,
      wallMs: Number(wallMs.toFixed(2)),
      prefillMs: Number((stats.prefillTimeMs ?? 0).toFixed(2)),
      ttftMs: Number((stats.ttftMs ?? 0).toFixed(2)),
      decodeMs: Number((stats.decodeTimeMs ?? 0).toFixed(2)),
      promptTokens: payload.usage.prompt_tokens,
      cachedTokens: payload.usage.prompt_tokens_details?.cached_tokens ?? 0,
      prefillComputedTokens: stats.prefillComputedTokens ?? null,
      completionTokens: payload.usage.completion_tokens,
      outputText: content,
      generatedTokenIds: payload.doppler_receipt?.output?.tokenIds ?? payload.doppler_receipt?.evidence?.tokenIds ?? [],
      prefixCacheHealth: health.prefix_cache ?? null,
    });
  }
  const finalHealth = await (await fetch(`http://127.0.0.1:${port}/health`)).json();
  return { turnMetrics, finalHealth, backendIdentity: turnMetrics[0] };
}

async function runArm(port, enableCache) {
  const { proc } = await startServer(port, enableCache);
  try {
    // Warm shader compilation once on an unrelated short prompt so follow-up comparison measures prefill vs cache, not cold shader compile.
    const warmRes = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: modelId,
        messages: [{ role: 'user', content: 'Warmup ping.' }],
        max_tokens: 2,
        temperature: 0,
        top_k: 1,
        include_receipt: true,
      }),
    });
    assert.equal(warmRes.status, 200);
    const warmPayload = await warmRes.json();
    const adapter = warmPayload.doppler_receipt?.backendIdentity?.adapter
      ?? warmPayload.doppler_receipt?.evidence?.backendIdentity?.adapter ?? null;
    const conv = await runConversation(port);
    return { enableCache, adapter, ...conv };
  } finally {
    proc.kill('SIGTERM');
    await new Promise((resolve) => proc.once('exit', resolve));
  }
}

const offArm = await runArm(18931, false);
const onArm = await runArm(18932, true);

for (let i = 0; i < TURNS.length; i += 1) {
  assert.equal(
    onArm.turnMetrics[i].outputText,
    offArm.turnMetrics[i].outputText,
    `Exact output mismatch on turn ${i + 1}: cache-on=${JSON.stringify(onArm.turnMetrics[i].outputText)} vs cache-off=${JSON.stringify(offArm.turnMetrics[i].outputText)}`
  );
  assert.deepEqual(
    onArm.turnMetrics[i].generatedTokenIds,
    offArm.turnMetrics[i].generatedTokenIds,
    `Exact token ID mismatch on turn ${i + 1}`
  );
}

const followUpsOffPrefillMs = offArm.turnMetrics.slice(1).reduce((s, t) => s + t.prefillMs, 0);
const followUpsOnPrefillMs = onArm.turnMetrics.slice(1).reduce((s, t) => s + t.prefillMs, 0);
const followUpsOffWallMs = offArm.turnMetrics.slice(1).reduce((s, t) => s + t.wallMs, 0);
const followUpsOnWallMs = onArm.turnMetrics.slice(1).reduce((s, t) => s + t.wallMs, 0);

const summary = {
  schema: 'dopplerx.serve-prefix-reuse-benchmark/v1',
  measuredUtc: new Date().toISOString(),
  modelId,
  adapter: onArm.adapter,
  exactTokenParityAllTurns: true,
  followUpSummary: {
    followUpTurns: TURNS.length - 1,
    cacheOffPrefillMs: Number(followUpsOffPrefillMs.toFixed(2)),
    cacheOnPrefillMs: Number(followUpsOnPrefillMs.toFixed(2)),
    savedPrefillMs: Number((followUpsOffPrefillMs - followUpsOnPrefillMs).toFixed(2)),
    prefillSpeedupFactor: followUpsOnPrefillMs > 0 ? Number((followUpsOffPrefillMs / followUpsOnPrefillMs).toFixed(2)) : null,
    cacheOffWallMs: Number(followUpsOffWallMs.toFixed(2)),
    cacheOnWallMs: Number(followUpsOnWallMs.toFixed(2)),
    savedWallMs: Number((followUpsOffWallMs - followUpsOnWallMs).toFixed(2)),
    retainedMemoryBytes: onArm.finalHealth.prefix_cache?.allocatedBytes ?? null,
    retainedEntries: onArm.finalHealth.prefix_cache?.entries ?? null,
    cacheHits: onArm.finalHealth.prefix_cache?.hits ?? null,
    cacheMisses: onArm.finalHealth.prefix_cache?.misses ?? null,
    cachedTokensInPool: onArm.finalHealth.prefix_cache?.cachedTokens ?? null,
    reusedTokensByTurn: onArm.turnMetrics.map((t) => t.cachedTokens),
  },
  cacheOffTurns: offArm.turnMetrics,
  cacheOnTurns: onArm.turnMetrics,
};

writeFileSync(outputPath, JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify(summary.followUpSummary, null, 2));
