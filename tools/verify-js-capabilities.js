#!/usr/bin/env node
import assert from 'node:assert/strict';
import http from 'node:http';
import { createReadStream, existsSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { getCachedModel } from '../src/client/doppler-api.js';
import { validateChoiceScoringResult } from '../src/config/choice-scoring.js';
import { getBufferPool } from '../src/memory/buffer-pool.js';
import { browserLaunchArgs, launchBrowser } from '../src/tooling/node-browser/transport.js';
import {
  applyChoiceTemperatureCalibration,
  evaluateChoiceCalibration,
  fitChoiceTemperatureCalibration,
} from '../src/inference/choice-calibration.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const [genUrl, embedUrl, rerankUrl, outputPath] = process.argv.slice(2);
if (!genUrl || !embedUrl || !rerankUrl || !outputPath) {
  console.error('Usage: node tools/verify-js-capabilities.js <genUrl> <embedUrl> <rerankUrl> <output.json>');
  process.exit(2);
}

const CHOICE_FIXTURE = JSON.parse(
  readFileSync(path.join(REPO_ROOT, 'tests/fixtures/choice-scoring-relevance-qwen.json'), 'utf8')
);
const SAMPLE_DOCS = ['sourdough.md', 'tire.txt', 'eclipse.md', 'solar.txt', 'git.md', 'starter-motor.txt'].map((name) => ({
  id: name,
  text: readFileSync(path.join(REPO_ROOT, 'examples/document-search/samples', name), 'utf8').trim(),
}));
const SEARCH_QUERIES = [
  { query: 'How does wild yeast fermentation work in sourdough bread?', expectedDoc: 'sourdough.md' },
  { query: 'How do I safely change a flat tire on the roadside?', expectedDoc: 'tire.txt' },
  { query: 'Why should I never view a solar eclipse without certified eye protection?', expectedDoc: 'eclipse.md' },
  { query: 'How do photovoltaic solar panels convert sunlight into electricity?', expectedDoc: 'solar.txt' },
  { query: 'How do Git commits and branches track repository history?', expectedDoc: 'git.md' },
  { query: 'Why does a car click without cranking when the starter motor solenoid fails?', expectedDoc: 'starter-motor.txt' },
];

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function snapshotNodeMemory() {
  const mem = process.memoryUsage();
  let gpu = null;
  try {
    const stats = getBufferPool().getStats();
    gpu = {
      activeBuffers: stats.activeBuffers,
      pooledBuffers: stats.pooledBuffers,
      currentBytesAllocated: stats.currentBytesAllocated,
      resources: stats.resources,
    };
  } catch {
    gpu = null;
  }
  return {
    rssBytes: mem.rss,
    heapUsedBytes: mem.heapUsed,
    externalBytes: mem.external,
    gpu,
  };
}

async function qualifyNodeCapabilities() {
  // 1. Local text generation: cold open, warm decode, mid-stream abort settlement, reuse, unload cleanup.
  const genOpenStart = performance.now();
  const genHandle = await getCachedModel({ registryId: 'qwen-3-5-0-8b-q4k-ef16-af32-vf16', url: genUrl });
  const genColdOpenMs = Number((performance.now() - genOpenStart).toFixed(2));
  const genResidentMem = snapshotNodeMemory();

  const genWarmStart = performance.now();
  const genFirst = await genHandle.chatText(
    [{ role: 'user', content: 'State the chemical formula of water in one word.' }],
    { maxTokens: 8, temperature: 0, topK: 1 }
  );
  const genFirstMs = Number((performance.now() - genWarmStart).toFixed(2));

  const abortCtrl = new AbortController();
  let streamedChunks = 0;
  let abortedCleanly = false;
  const cancelStart = performance.now();
  try {
    for await (const chunk of genHandle.chat(
      [{ role: 'user', content: 'Count slowly from one to fifty in English words.' }],
      { maxTokens: 32, temperature: 0, topK: 1, signal: abortCtrl.signal }
    )) {
      if (chunk) streamedChunks += 1;
      if (streamedChunks >= 2) abortCtrl.abort(new Error('caller-cancelled'));
    }
  } catch {
    abortedCleanly = true;
  }
  const genCancelSettleMs = Number((performance.now() - cancelStart).toFixed(2));
  assert.equal(abortedCleanly, true, 'Mid-stream generation abort must reject cleanly');

  const genPostCancelStart = performance.now();
  const genAfterCancel = await genHandle.chatText(
    [{ role: 'user', content: 'State the chemical formula of water in one word.' }],
    { maxTokens: 8, temperature: 0, topK: 1 }
  );
  const genPostCancelMs = Number((performance.now() - genPostCancelStart).toFixed(2));
  assert.equal(genAfterCancel.content, genFirst.content, 'Post-cancellation generation must match deterministic baseline');

  const genUnloadStart = performance.now();
  await genHandle.unload();
  const genUnloadMs = Number((performance.now() - genUnloadStart).toFixed(2));
  const genAfterUnloadMem = snapshotNodeMemory();
  assert.equal(genAfterUnloadMem.gpu?.activeBuffers ?? 0, 0, 'Generation unload must leave 0 active GPU buffers');

  // 2. Embeddings + Reranking dual-resident local document search + independent lifecycle & cancellation.
  const embOpenStart = performance.now();
  const embHandle = await getCachedModel({ url: embedUrl });
  const embColdOpenMs = Number((performance.now() - embOpenStart).toFixed(2));

  const preAbort = new AbortController();
  preAbort.abort();
  let embAborted = false;
  try {
    await embHandle.embed('Aborted embedding probe', { signal: preAbort.signal });
  } catch {
    embAborted = true;
  }
  assert.equal(embAborted, true, 'Pre-aborted embedding call must reject cleanly');

  const indexStart = performance.now();
  const docEmbeddings = [];
  for (const doc of SAMPLE_DOCS) {
    const ev = await embHandle.embedWithEvidence(doc.text);
    const vec = ev.embedding ?? ev.vector;
    assert.equal(vec.length, 1024, 'Expected 1024-dim Qwen3 embedding');
    docEmbeddings.push(vec);
  }
  const indexSixDocsMs = Number((performance.now() - indexStart).toFixed(2));

  const rerOpenStart = performance.now();
  const rerHandle = await getCachedModel({ url: rerankUrl });
  const rerColdOpenMs = Number((performance.now() - rerOpenStart).toFixed(2));
  const dualResidentMem = snapshotNodeMemory();

  let rerAborted = false;
  try {
    await rerHandle.rerankWithEvidence('Aborted rerank probe', ['doc a', 'doc b'], { signal: preAbort.signal });
  } catch {
    rerAborted = true;
  }
  assert.equal(rerAborted, true, 'Pre-aborted rerank call must reject cleanly');

  const searchResults = [];
  for (const item of SEARCH_QUERIES) {
    const qStart = performance.now();
    const qEmb = await embHandle.embedWithEvidence(item.query);
    const qVec = qEmb.embedding ?? qEmb.vector;
    const scored = SAMPLE_DOCS.map((doc, idx) => ({
      id: doc.id,
      text: doc.text,
      cosine: cosineSimilarity(qVec, docEmbeddings[idx]),
    })).sort((a, b) => b.cosine - a.cosine);
    const candidates = scored.slice(0, 3);
    const reranked = await rerHandle.rerankWithEvidence(
      item.query,
      candidates.map((c) => c.text)
    );
    const topCandidate = candidates[reranked.ranking[0].index];
    const queryMs = Number((performance.now() - qStart).toFixed(2));
    assert.equal(topCandidate.id, item.expectedDoc, `Expected top document ${item.expectedDoc} for query "${item.query}", got ${topCandidate.id}`);
    searchResults.push({
      expectedDoc: item.expectedDoc,
      selectedDoc: topCandidate.id,
      topCosine: Number(topCandidate.cosine.toFixed(4)),
      topRerankScore: Number(reranked.ranking[0].score.toFixed(4)),
      queryMs,
    });
  }

  await embHandle.unload();
  const afterEmbUnloadMem = snapshotNodeMemory();
  assert.ok((afterEmbUnloadMem.gpu?.activeBuffers ?? 0) > 0, 'Reranker buffers remain resident after unloading embedding model');

  // 3. Typed Choice Scoring + Empirical Temperature Calibration on physical Metal logits.
  const choiceStart = performance.now();
  const scoredCases = [];
  for (const testCase of CHOICE_FIXTURE.cases) {
    const request = {
      prompt: `${CHOICE_FIXTURE.prefix}${testCase.question}${CHOICE_FIXTURE.suffix}`,
      choices: CHOICE_FIXTURE.choices,
      maxSeqLen: CHOICE_FIXTURE.maxSeqLen,
    };
    const raw = validateChoiceScoringResult(request, await rerHandle.scoreChoices(request));
    assert.equal(raw.selectedId, testCase.expectedId, `Choice case ${testCase.id} expected ${testCase.expectedId}`);
    scoredCases.push({
      id: testCase.id,
      targetId: testCase.expectedId,
      raw,
    });
  }
  const choiceTwelveCasesMs = Number((performance.now() - choiceStart).toFixed(2));

  let choiceAborted = false;
  try {
    await rerHandle.scoreChoices({
      prompt: `${CHOICE_FIXTURE.prefix}${CHOICE_FIXTURE.cases[0].question}${CHOICE_FIXTURE.suffix}`,
      choices: CHOICE_FIXTURE.choices,
      maxSeqLen: CHOICE_FIXTURE.maxSeqLen,
    }, { signal: preAbort.signal });
  } catch {
    choiceAborted = true;
  }
  assert.equal(choiceAborted, true, 'Aborted scoreChoices must reject and reset continuation cleanly');

  const fitSplit = scoredCases.slice(0, 6).map((c) => ({ result: c.raw, targetId: c.targetId }));
  const evalSplit = scoredCases.slice(6).map((c) => ({ result: c.raw, targetId: c.targetId }));
  const tempFit = fitChoiceTemperatureCalibration(fitSplit, {
    datasetId: 'choice-scoring-relevance-qwen:train-0..5',
  });
  const heldOutUncalibrated = evaluateChoiceCalibration(evalSplit, { temperature: 1 });
  const heldOutCalibrated = evaluateChoiceCalibration(evalSplit, { temperature: tempFit.temperature });
  const sampleCalibrated = applyChoiceTemperatureCalibration(scoredCases[6].raw, tempFit);

  const rerUnloadStart = performance.now();
  await rerHandle.unload();
  const rerUnloadMs = Number((performance.now() - rerUnloadStart).toFixed(2));
  const finalMem = snapshotNodeMemory();
  assert.equal(finalMem.gpu?.activeBuffers ?? 0, 0, 'Final unload must leave 0 active GPU buffers');

  return {
    backendIdentity: genFirst.evidence?.backendIdentity ?? null,
    generation: {
      modelId: 'qwen-3-5-0-8b-q4k-ef16-af32-vf16',
      coldOpenMs: genColdOpenMs,
      firstRequestMs: genFirstMs,
      midStreamAbortChunksObserved: streamedChunks,
      cancelSettleMs: genCancelSettleMs,
      postCancelRequestMs: genPostCancelMs,
      postCancelExactMatch: true,
      unloadMs: genUnloadMs,
      residentMemory: genResidentMem,
      afterUnloadMemory: genAfterUnloadMem,
    },
    embeddingAndDualResidentSearch: {
      embeddingModelId: 'qwen-3-embedding-0-6b-q4k-ehf16-af32',
      rerankerModelId: 'qwen-3-reranker-0-6b-f16-af32',
      embeddingColdOpenMs: embColdOpenMs,
      rerankerColdOpenMs: rerColdOpenMs,
      indexSixSampleDocsMs: indexSixDocsMs,
      queriesTested: searchResults.length,
      exactTop1Matches: searchResults.length,
      dualResidentMemory: dualResidentMem,
      searchResults,
    },
    choiceScoringAndCalibration: {
      modelId: 'qwen-3-reranker-0-6b-f16-af32',
      casesEvaluated: scoredCases.length,
      correctSelections: scoredCases.length,
      twelveCasesWallMs: choiceTwelveCasesMs,
      cancellationVerified: choiceAborted,
      rerankerUnloadMs: rerUnloadMs,
      afterUnloadMemory: finalMem,
      temperatureCalibration: {
        fitDatasetId: tempFit.datasetId,
        fittedTemperature: Number(tempFit.temperature.toFixed(6)),
        trainBeforeLogLoss: Number(tempFit.before.meanLogLoss.toFixed(6)),
        trainAfterLogLoss: Number(tempFit.after.meanLogLoss.toFixed(6)),
        heldOutCount: heldOutCalibrated.sampleCount,
        heldOutAccuracy: heldOutCalibrated.accuracy,
        heldOutUncalibratedLogLoss: Number(heldOutUncalibrated.meanLogLoss.toFixed(6)),
        heldOutCalibratedLogLoss: Number(heldOutCalibrated.meanLogLoss.toFixed(6)),
        heldOutUncalibratedBrier: Number(heldOutUncalibrated.brierScore.toFixed(6)),
        heldOutCalibratedBrier: Number(heldOutCalibrated.brierScore.toFixed(6)),
        heldOutUncalibratedEce: Number(heldOutUncalibrated.ece.toFixed(6)),
        heldOutCalibratedEce: Number(heldOutCalibrated.ece.toFixed(6)),
        sampleCalibratedSchema: sampleCalibrated.schema,
        sampleCalibratedProbabilities: sampleCalibrated.choices.map((c) => ({
          id: c.id,
          logit: Number(c.logit.toFixed(4)),
          probability: Number(c.probability.toFixed(6)),
        })),
      },
    },
  };
}

function resolveLocalDir(fileUrl) {
  return fileURLToPath(new URL(fileUrl));
}

async function qualifyBrowserCapabilities() {
  const { chromium } = await import('playwright');
  const mounts = {
    '/models/gen/': resolveLocalDir(genUrl.endsWith('/') ? genUrl : `${genUrl}/`),
    '/models/embed/': resolveLocalDir(embedUrl.endsWith('/') ? embedUrl : `${embedUrl}/`),
    '/models/rerank/': resolveLocalDir(rerankUrl.endsWith('/') ? rerankUrl : `${rerankUrl}/`),
  };
  const mimeByExt = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.wgsl': 'text/plain; charset=utf-8',
    '.bin': 'application/octet-stream',
  };

  const server = http.createServer((req, res) => {
    const reqUrl = new URL(req.url, 'http://127.0.0.1');
    if (reqUrl.pathname === '/index.html') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end('<!doctype html><meta charset="utf-8"><title>Doppler Browser Capabilities</title>');
      return;
    }
    let filePath = null;
    for (const [prefix, dir] of Object.entries(mounts)) {
      if (reqUrl.pathname.startsWith(prefix)) {
        filePath = path.join(dir, reqUrl.pathname.slice(prefix.length));
        break;
      }
    }
    if (!filePath) {
      filePath = path.join(REPO_ROOT, reqUrl.pathname.slice(1));
    }
    if (!existsSync(filePath) || !statSync(filePath).isFile()) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }
    const stat = statSync(filePath);
    const mime = mimeByExt[path.extname(filePath)] ?? 'application/octet-stream';
    const rangeHeader = req.headers.range;
    if (rangeHeader) {
      const m = /^bytes=(\d+)-(\d*)$/.exec(rangeHeader);
      if (m) {
        const start = Number(m[1]);
        const end = m[2] ? Math.min(Number(m[2]), stat.size - 1) : stat.size - 1;
        res.writeHead(206, {
          'Content-Type': mime,
          'Content-Length': end - start + 1,
          'Content-Range': `bytes ${start}-${end}/${stat.size}`,
          'Accept-Ranges': 'bytes',
        });
        createReadStream(filePath, { start, end }).pipe(res);
        return;
      }
    }
    res.writeHead(200, {
      'Content-Type': mime,
      'Content-Length': stat.size,
      'Accept-Ranges': 'bytes',
    });
    createReadStream(filePath).pipe(res);
  });

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  const baseOrigin = `http://127.0.0.1:${port}`;

  const browser = await launchBrowser(chromium, {
    headless: true,
    args: browserLaunchArgs(),
  });
  try {
    const page = await browser.newPage();
    await page.goto(`${baseOrigin}/index.html`);
    const browserResult = await page.evaluate(async ({ choicePrompt, choiceItems }) => {
      if (!navigator.gpu) {
        return { available: false, reason: 'navigator.gpu unavailable' };
      }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        return { available: false, reason: 'requestAdapter returned null' };
      }
      const adapterInfo = adapter.info ? {
        vendor: adapter.info.vendor || null,
        architecture: adapter.info.architecture || null,
        description: adapter.info.description || null,
      } : null;

      const { getCachedModel } = await import('/src/client/doppler-api.js');
      const { getBufferPool } = await import('/src/memory/buffer-pool.js');
      const { applyChoiceTemperatureCalibration } = await import('/src/inference/choice-calibration.js');

      const origin = window.location.origin;

      // 1. Browser Generation + Cancellation + Unload
      const tGen0 = performance.now();
      const gen = await getCachedModel({
        registryId: 'qwen-3-5-0-8b-q4k-ef16-af32-vf16',
        url: `${origin}/models/gen`,
      });
      const genOpenMs = Number((performance.now() - tGen0).toFixed(2));
      const tGenRun = performance.now();
      const genOut = await gen.chatText(
        [{ role: 'user', content: 'State the chemical formula of water in one word.' }],
        { maxTokens: 8, temperature: 0, topK: 1 }
      );
      const genRunMs = Number((performance.now() - tGenRun).toFixed(2));
      const abortCtrl = new AbortController();
      abortCtrl.abort();
      let genCancelled = false;
      try {
        await gen.chatText([{ role: 'user', content: 'Cancelled' }], { signal: abortCtrl.signal });
      } catch {
        genCancelled = true;
      }
      await gen.unload();
      const afterGenActive = getBufferPool().getStats().activeBuffers;

      // 2. Browser Embedding + Unload
      const tEmb0 = performance.now();
      const emb = await getCachedModel({ url: `${origin}/models/embed` });
      const embOpenMs = Number((performance.now() - tEmb0).toFixed(2));
      const embQ = await emb.embedWithEvidence('What is the capital of France?');
      const embPos = await emb.embedWithEvidence('Paris is the capital of France.');
      const embNeg = await emb.embedWithEvidence('Sourdough bread uses wild yeast.');
      const vQ = embQ.embedding ?? embQ.vector;
      const vPos = embPos.embedding ?? embPos.vector;
      const vNeg = embNeg.embedding ?? embNeg.vector;
      const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
      const cosPos = dot(vQ, vPos);
      const cosNeg = dot(vQ, vNeg);
      await emb.unload();
      const afterEmbActive = getBufferPool().getStats().activeBuffers;

      // 3. Browser Reranking + Typed Choice Scoring + Calibration + Unload
      const tRer0 = performance.now();
      const rer = await getCachedModel({ url: `${origin}/models/rerank` });
      const rerOpenMs = Number((performance.now() - tRer0).toFixed(2));
      const rerOut = await rer.rerankWithEvidence(
        'What is the capital of France?',
        ['Berlin is the capital of Germany.', 'Paris is the capital of France.']
      );
      const rawChoice = await rer.scoreChoices({
        prompt: choicePrompt,
        choices: choiceItems,
        maxSeqLen: 512,
      });
      const calibratedChoice = applyChoiceTemperatureCalibration(rawChoice, {
        schema: 'doppler.choice-calibration/v1',
        method: 'temperature-scaling',
        temperature: 1.25,
        fittedOnCount: 6,
        datasetId: 'browser-smoke',
      });
      await rer.unload();
      const afterRerActive = getBufferPool().getStats().activeBuffers;

      return {
        available: true,
        userAgent: navigator.userAgent,
        adapterInfo,
        generation: {
          openMs: genOpenMs,
          runMs: genRunMs,
          outputLength: genOut.content.length,
          cancelledCleanly: genCancelled,
          activeBuffersAfterUnload: afterGenActive,
        },
        embedding: {
          openMs: embOpenMs,
          dimension: vQ.length,
          positiveCosine: Number(cosPos.toFixed(4)),
          negativeCosine: Number(cosNeg.toFixed(4)),
          positiveExceedsNegative: cosPos > cosNeg,
          activeBuffersAfterUnload: afterEmbActive,
        },
        rerankAndChoiceScoring: {
          openMs: rerOpenMs,
          topRankedIndex: rerOut.ranking[0].index,
          choiceSelectedId: rawChoice.selectedId,
          calibratedSelectedProbability: Number(calibratedChoice.choices[0].probability.toFixed(6)),
          activeBuffersAfterUnload: afterRerActive,
        },
      };
    }, {
      choicePrompt: `${CHOICE_FIXTURE.prefix}${CHOICE_FIXTURE.cases[0].question}${CHOICE_FIXTURE.suffix}`,
      choiceItems: CHOICE_FIXTURE.choices,
    });
    return browserResult;
  } finally {
    await browser.close();
    server.close();
  }
}

const nodeResult = await qualifyNodeCapabilities();
const browserResult = await qualifyBrowserCapabilities();

assert.equal(browserResult.available, true, `Browser WebGPU must succeed: ${browserResult.reason ?? ''}`);
assert.equal(browserResult.generation.activeBuffersAfterUnload, 0);
assert.equal(browserResult.embedding.positiveExceedsNegative, true);
assert.equal(browserResult.embedding.activeBuffersAfterUnload, 0);
assert.equal(browserResult.rerankAndChoiceScoring.topRankedIndex, 1);
assert.equal(browserResult.rerankAndChoiceScoring.choiceSelectedId, 'relevant');
assert.equal(browserResult.rerankAndChoiceScoring.activeBuffersAfterUnload, 0);

const receipt = {
  schema: 'dopplerx.js-capabilities-qualification/v1',
  qualifiedUtc: new Date().toISOString(),
  packageName: 'dopplerx',
  packageVersion: '0.1.5',
  nodeSurface: nodeResult,
  browserSurface: browserResult,
};

const serialized = JSON.stringify(receipt, null, 2) + '\n';
if (/\/Users\/|\/home\//.test(serialized)) {
  throw new Error('Refusing to write receipt containing private host paths.');
}
writeFileSync(outputPath, serialized);
console.log(JSON.stringify({
  nodeExactSearchMatches: nodeResult.embeddingAndDualResidentSearch.exactTop1Matches,
  nodeChoiceCorrect: nodeResult.choiceScoringAndCalibration.correctSelections,
  temperatureCalibration: nodeResult.choiceScoringAndCalibration.temperatureCalibration,
  browserSurfaceAvailable: browserResult.available,
  browserAdapter: browserResult.adapterInfo,
}, null, 2));
process.exit(0);
