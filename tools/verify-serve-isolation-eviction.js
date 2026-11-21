#!/usr/bin/env node
/**
 * Physical GPU verification of standalone loopback `doppler-serve` multi-conversation
 * isolation, bounded LRU byte-pool eviction, and safe post-eviction session reuse.
 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const [modelId, modelUrl, outputPath] = process.argv.slice(2);
if (!modelId || !modelUrl || !outputPath) {
  console.error('Usage: node tools/verify-serve-isolation-eviction.js <modelId> <modelUrl> <output.json>');
  process.exit(2);
}

// Each Qwen 0.8B hybrid context occupies ~146.7 MB (96 MiB KV + 43.91 MiB Gated DeltaNet).
// A 320 MiB (335,544,320 byte) pool fits AT MOST 2 concurrent conversations (293.4 MB) and MUST evict on a 3rd!
const BOUNDED_TWO_ENTRY_BYTES = 320 * 1024 * 1024;

const CONV_ALPHA = {
  id: 'alpha',
  system: 'You are session ALPHA. Always state only the exact requested fact in at most eight words. Reference code: ALPHA-739.',
  turns: [
    'What is your session name and reference code?',
    'Repeat only the three digits of your reference code.',
    'State the chemical symbol for gold.',
  ],
};

const CONV_BRAVO = {
  id: 'bravo',
  system: 'You are session BRAVO. Always state only the exact requested fact in at most eight words. Reference code: BRAVO-412.',
  turns: [
    'What is your session name and reference code?',
    'Repeat only the three digits of your reference code.',
  ],
};

const CONV_CHARLIE = {
  id: 'charlie',
  system: 'You are session CHARLIE. Always state only the exact requested fact in at most eight words. Reference code: CHARLIE-905.',
  turns: [
    'What is your session name and reference code?',
    'Repeat only the three digits of your reference code.',
  ],
};

async function startServe(port, enableCache, maxBytes = BOUNDED_TWO_ENTRY_BYTES) {
  const args = [
    path.join(REPO_ROOT, 'src/cli/doppler-serve.js'),
    '--host', '127.0.0.1',
    '--port', String(port),
    '--model', modelId,
    '--model-url', modelUrl,
    ...(enableCache ? ['--prefix-cache', '--prefix-cache-bytes', String(maxBytes)] : ['--no-prefix-cache']),
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
        if (body.status === 'ok' || body.ok === true) return { proc };
      }
    } catch {
      // Wait for listen.
    }
    await new Promise((r) => setTimeout(r, 150));
  }
  proc.kill('SIGTERM');
  throw new Error(`Timed out waiting for doppler-serve on port ${port}: ${stderr}`);
}

async function sendChat(port, messages) {
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
  const wallMs = Number((performance.now() - t0).toFixed(2));
  assert.equal(res.status, 200, `Expected HTTP 200, got ${res.status}`);
  const payload = await res.json();
  const health = await (await fetch(`http://127.0.0.1:${port}/health`)).json();
  const content = payload.choices[0].message.content;
  const tokenIds = payload.doppler_receipt?.output?.tokenIds ?? payload.doppler_receipt?.evidence?.tokenIds ?? [];
  const adapter = payload.doppler_receipt?.backendIdentity?.adapter ?? null;
  const cachedTokens = payload.usage.prompt_tokens_details?.cached_tokens ?? 0;
  const prefillMs = Number((payload.doppler_receipt?.timing?.prefillTimeMs ?? 0).toFixed(2));
  return {
    content,
    tokenIds,
    adapter,
    wallMs,
    prefillMs,
    promptTokens: payload.usage.prompt_tokens,
    cachedTokens,
    prefixCache: health.prefix_cache ?? null,
  };
}

// 1. Collect uncached ground-truth reference outputs for Alpha, Bravo, and Charlie.
const baseline = {};
{
  const { proc } = await startServe(18941, false);
  try {
    for (const spec of [CONV_ALPHA, CONV_BRAVO, CONV_CHARLIE]) {
      const history = [{ role: 'system', content: spec.system }];
      baseline[spec.id] = [];
      for (const userPrompt of spec.turns) {
        history.push({ role: 'user', content: userPrompt });
        const step = await sendChat(18941, history);
        history.push({ role: 'assistant', content: step.content });
        baseline[spec.id].push(step);
      }
    }
  } finally {
    proc.kill('SIGTERM');
    await new Promise((resolve) => proc.once('exit', resolve));
  }
}

// 2. Run interleaved multi-conversation isolation + bounded 320 MiB LRU eviction + post-eviction reuse on integrated doppler-serve.
const timeline = [];
let adapter = null;
let finalHealth = null;
{
  const { proc } = await startServe(18942, true, BOUNDED_TWO_ENTRY_BYTES);
  try {
    const histories = {
      alpha: [{ role: 'system', content: CONV_ALPHA.system }],
      bravo: [{ role: 'system', content: CONV_BRAVO.system }],
      charlie: [{ role: 'system', content: CONV_CHARLIE.system }],
    };
    const schedule = [
      { conv: 'alpha', turnIdx: 0, expectHit: false, expectEvictions: 0, expectEntries: 1 },
      { conv: 'bravo', turnIdx: 0, expectHit: false, expectEvictions: 0, expectEntries: 2 },
      // Interleaved follow-ups while BOTH Alpha and Bravo reside in the 320 MiB pool:
      { conv: 'alpha', turnIdx: 1, expectHit: true, expectEvictions: 0, expectEntries: 2 },
      { conv: 'bravo', turnIdx: 1, expectHit: true, expectEvictions: 0, expectEntries: 2 },
      // Charlie arrives: 3rd context (~146.7 MB) exceeds 320 MiB pool -> MUST evict LRU (Alpha -> evictions = 1)!
      { conv: 'charlie', turnIdx: 0, expectHit: false, expectEvictions: 1, expectEntries: 2 },
      // Charlie follow-up hits its retained prefix while Bravo is still resident:
      { conv: 'charlie', turnIdx: 1, expectHit: true, expectEvictions: 1, expectEntries: 2 },
      // Evicted Alpha returns for Turn 3 -> safe cache miss (0 cached tokens), exact parity with uncached reference, evicts Bravo (evictions = 2)!
      { conv: 'alpha', turnIdx: 2, expectHit: false, expectEvictions: 2, expectEntries: 2 },
    ];

    for (const item of schedule) {
      const spec = item.conv === 'alpha' ? CONV_ALPHA : item.conv === 'bravo' ? CONV_BRAVO : CONV_CHARLIE;
      const hist = histories[item.conv];
      hist.push({ role: 'user', content: spec.turns[item.turnIdx] });
      const res = await sendChat(18942, hist);
      hist.push({ role: 'assistant', content: res.content });
      adapter = adapter || res.adapter;

      const ref = baseline[item.conv][item.turnIdx];
      assert.equal(
        res.content,
        ref.content,
        `Isolation/eviction output mismatch on ${item.conv} turn ${item.turnIdx + 1}: got ${JSON.stringify(res.content)} vs ref ${JSON.stringify(ref.content)}`
      );
      assert.deepEqual(
        res.tokenIds,
        ref.tokenIds,
        `Exact token ID mismatch on ${item.conv} turn ${item.turnIdx + 1}`
      );
      if (item.expectHit) {
        assert.ok(res.cachedTokens > 0, `Expected prefix cache hit on ${item.conv} turn ${item.turnIdx + 1}`);
      } else {
        assert.equal(res.cachedTokens, 0, `Expected prefix cache miss on ${item.conv} turn ${item.turnIdx + 1}`);
      }
      assert.equal(
        res.prefixCache.evictions,
        item.expectEvictions,
        `Expected evictions=${item.expectEvictions} after ${item.conv} turn ${item.turnIdx + 1}, got ${res.prefixCache.evictions}`
      );
      assert.equal(
        res.prefixCache.entries,
        item.expectEntries,
        `Expected entries=${item.expectEntries} after ${item.conv} turn ${item.turnIdx + 1}, got ${res.prefixCache.entries}`
      );
      assert.ok(
        res.prefixCache.allocatedBytes <= BOUNDED_TWO_ENTRY_BYTES,
        `Pool allocatedBytes (${res.prefixCache.allocatedBytes}) exceeded maxBytes (${BOUNDED_TWO_ENTRY_BYTES})`
      );

      timeline.push({
        conversation: item.conv,
        turn: item.turnIdx + 1,
        cacheHit: res.cachedTokens > 0,
        cachedTokens: res.cachedTokens,
        promptTokens: res.promptTokens,
        prefillMs: res.prefillMs,
        wallMs: res.wallMs,
        exactParityWithUncachedBaseline: true,
        outputText: res.content,
        poolEntries: res.prefixCache.entries,
        poolAllocatedBytes: res.prefixCache.allocatedBytes,
        poolEvictions: res.prefixCache.evictions,
        poolHits: res.prefixCache.hits,
        poolMisses: res.prefixCache.misses,
      });
      finalHealth = res.prefixCache;
    }
  } finally {
    proc.kill('SIGTERM');
    await new Promise((resolve) => proc.once('exit', resolve));
  }
}

const receipt = {
  schema: 'dopplerx.serve-isolation-and-eviction/v1',
  verifiedUtc: new Date().toISOString(),
  modelId,
  adapter,
  poolMaxBytes: BOUNDED_TWO_ENTRY_BYTES,
  interleavedIsolationVerified: true,
  crossConversationTokenLeakage: 0,
  boundedLruEvictionVerified: true,
  postEvictionSessionReuseExactParity: true,
  finalPoolStats: finalHealth,
  interleavedTimeline: timeline,
};

writeFileSync(outputPath, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify({
  interleavedIsolationVerified: receipt.interleavedIsolationVerified,
  boundedLruEvictionVerified: receipt.boundedLruEvictionVerified,
  postEvictionSessionReuseExactParity: receipt.postEvictionSessionReuseExactParity,
  finalPoolStats: receipt.finalPoolStats,
}, null, 2));
