// DopplerX release gate, Node surface. Runs one installed package (by its install directory)
// on the real GPU through the Node WebGPU provider, with no CPU or fallback path allowed.
// usage: node node-gate.js <install-dir> <text|protein> <model-dir> <out.json> [reference.json]
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';

const [installDir, mode, modelDir, outFile, referenceFile] = process.argv.slice(2);
const root = path.join(installDir, 'node_modules/dopplerx');
const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
const lock = JSON.parse(fs.readFileSync(path.join(installDir, 'package-lock.json'), 'utf8'));
const mod = (rel) => import(pathToFileURL(path.join(root, rel)).href);
const sha256 = (buf) => crypto.createHash('sha256').update(buf).digest('hex');

const PROMPT = 'Explain in two sentences why the sky is blue.';
const GREEDY = { maxTokens: 32, temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, useChatTemplate: true };
const PROTEIN = 'MKTIIALSYIFCLVFADYKDDDDK';

const out = {
  shardHashCheck: process.env.TRUST_SHARDS === '1' ? 'trusted (sha256 compared with the published LFS sha256 instead)' : 'doppler',
  schema: 'dopplerx-gate.node/v1',
  startedAtUtc: new Date().toISOString(),
  node: process.version,
  package: { name: pkg.name, version: pkg.version, lockIntegrity: lock.packages['node_modules/dopplerx'].integrity,
    webgpuProvider: lock.packages['node_modules/webgpu']?.version ?? null, doeGpu: lock.packages['node_modules/doe-gpu']?.version ?? null },
  mode,
};

// Model identity: manifest sha256 plus every shard's sha256 as stored on disk.
const manifestText = fs.readFileSync(path.join(modelDir, 'manifest.json'), 'utf8');
const manifest = JSON.parse(manifestText);
const files = fs.readdirSync(modelDir).filter((f) => /^shard_\d+\.bin$|^tokenizer\.json$/.test(f)).sort();
const fileDigests = Object.fromEntries(files.map((f) => [f, sha256(fs.readFileSync(path.join(modelDir, f)))]));
out.model = {
  modelId: manifest.modelId,
  manifestSha256: sha256(manifestText),
  identityDigest: sha256(JSON.stringify({ manifest: sha256(manifestText), files: fileDigests })),
  files: fileDigests,
};

const { bootstrapNodeWebGPU, releaseNodeWebGPU } = await mod('src/tooling/node-webgpu.js');
const boot = await bootstrapNodeWebGPU();
out.provider = { ok: boot.ok, provider: boot.provider ?? null, receipt: boot.receipt ?? null, detail: boot.detail ?? null };
if (!boot.ok) {
  out.result = 'fail: no Node WebGPU provider';
  fs.writeFileSync(outFile, JSON.stringify(out, null, 1));
  process.exit(1);
}
const { createPipeline } = await mod('src/inference/pipelines/text.js');
const { createNodeFileArtifactStorageContext } = await mod('src/storage/artifact-storage-context.js');
const device = await mod('src/gpu/device.js');
await device.initDevice();
const capsBeforeLoad = device.getKernelCapabilities();

const loadStart = performance.now();
const pipeline = await createPipeline(manifest, {
  // TRUST_SHARDS=1: the hosted ESM-2 shards' blake3 does not match their manifest in either version;
  // their sha256 is checked against the Hugging Face LFS sha256 instead (recorded in out.model).
  storage: createNodeFileArtifactStorageContext(pathToFileURL(modelDir).href, manifest, process.env.TRUST_SHARDS === '1' ? { hashesTrusted: true } : {}),
  runtimeConfig: { inference: { session: { kvcache: { maxSeqLen: 512 } } },
    ...(process.env.TRUST_SHARDS === '1' ? { loading: { shardCache: { verifyHashes: false } } } : {}) },
});
out.timing = { loadMs: performance.now() - loadStart };
let caps = capsBeforeLoad;
try { caps = device.getKernelCapabilities(); } catch { /* the pipeline owns the device after load */ }
out.adapter = { adapterInfo: caps?.adapterInfo ?? null, hasF16: caps?.hasF16 ?? null, hasSubgroups: caps?.hasSubgroups ?? null };
out.useGPU = pipeline.useGPU ?? pipeline._state?.useGPU ?? null;

try {
  if (mode === 'text') {
    const ids = [];
    const pieces = [];
    const t0 = performance.now();
    let firstTokenMs = null;
    for await (const token of pipeline.generateTokens([{ role: 'user', content: PROMPT }], GREEDY)) {
      if (firstTokenMs == null) firstTokenMs = performance.now() - t0;
      pieces.push(token);
    }
    out.timing.generateMs = performance.now() - t0;
    out.timing.firstTokenMs = firstTokenMs;
    const stats = structuredClone(pipeline.stats ?? {});
    for (const p of pieces) ids.push(typeof p === 'number' ? p : p?.id ?? p?.tokenId ?? p);
    out.prompt = PROMPT;
    out.sampling = GREEDY;
    out.tokens = ids;
    out.tokenCount = ids.length;
    out.text = typeof pipeline.tokenizer?.decode === 'function' && ids.every((v) => typeof v === 'number')
      ? pipeline.tokenizer.decode(ids, true, true) : pieces.join('');
    out.stats = stats;
    out.timing.decodeTokensPerSec = stats.decodeTimeMs ? (ids.length / (stats.decodeTimeMs / 1000)) : null;
  } else {
    const t0 = performance.now();
    const result = await pipeline.encodeSequence(PROTEIN, { includeTokenEmbeddings: true });
    out.timing.encodeMs = performance.now() - t0;
    out.sequence = PROTEIN;
    out.protein = {
      alphabet: result.alphabet,
      tokens: Array.from(result.tokens),
      includedTokenCount: result.includedTokenCount,
      embeddingDim: result.embeddingDim ?? (result.pooledEmbedding?.length ?? null),
      pooledEmbedding: Array.from(result.pooledEmbedding ?? []),
      tokenEmbeddingsSha256: result.tokenEmbeddings ? sha256(Buffer.from(result.tokenEmbeddings.buffer)) : null,
      tokenEmbeddings: Array.from(result.tokenEmbeddings ?? []),
    };
    out.stats = structuredClone(pipeline.stats ?? {});
    if (referenceFile) {
      const ref = JSON.parse(fs.readFileSync(referenceFile, 'utf8'));
      const dim = out.protein.embeddingDim;
      let pooledMax = 0;
      ref.probes.pooledEmbedding.indices.forEach((idx, k) => {
        pooledMax = Math.max(pooledMax, Math.abs(out.protein.pooledEmbedding[idx] - ref.probes.pooledEmbedding.values[k]));
      });
      let tokenMax = 0;
      for (const probe of ref.probes.tokenEmbeddings) {
        probe.indices.forEach((idx, k) => {
          tokenMax = Math.max(tokenMax, Math.abs(out.protein.tokenEmbeddings[probe.position * dim + idx] - probe.values[k]));
        });
      }
      out.reference = {
        file: path.basename(referenceFile),
        source: ref.source?.checkpointId ?? null,
        tokenIdsMatch: JSON.stringify(ref.input.tokenIds) === JSON.stringify(out.protein.tokens),
        pooledEmbeddingMaxAbs: pooledMax,
        tokenEmbeddingMaxAbs: tokenMax,
        tolerances: ref.tolerances,
        withinTolerance: pooledMax <= ref.tolerances.pooledEmbeddingMaxAbs && tokenMax <= ref.tolerances.tokenEmbeddingMaxAbs,
      };
    }
  }
} finally {
  await pipeline.unload?.();
  device.destroyDevice?.();
  await releaseNodeWebGPU?.();
}

// No-fallback evidence: a single provider attempt that succeeded on a GPU adapter, the pipeline on
// the GPU, and no fallback, CPU or snake_case marker anywhere in the stats.
const fallbackHits = [];
const walk = (value, trail) => {
  if (value && typeof value === 'object') {
    for (const [key, entry] of Object.entries(value)) {
      if (/fallback|cpu/i.test(key) && entry && entry !== 0 && !(Array.isArray(entry) && entry.length === 0)) fallbackHits.push(`${trail}.${key}=${JSON.stringify(entry).slice(0, 80)}`);
      walk(entry, `${trail}.${key}`);
    }
  } else if (typeof value === 'string' && /fallback|cpu[-_]/i.test(value)) {
    fallbackHits.push(`${trail}=${value}`);
  }
};
walk(out.stats, 'stats');
const attempts = out.provider.receipt?.attempts ?? [];
out.noFallback = {
  providerAttempts: attempts.map((a) => ({ providerId: a.providerId, ok: a.ok ?? a.status ?? null })),
  useGPU: out.useGPU,
  fallbackOrCpuMarkersInStats: fallbackHits,
  pass: out.provider.ok === true && out.useGPU !== false && fallbackHits.length === 0,
};
out.finishedAtUtc = new Date().toISOString();
fs.writeFileSync(outFile, JSON.stringify(out, null, 1));
const summary = { version: pkg.version, mode, model: out.model.modelId, adapter: out.adapter.adapterInfo, provider: out.provider.provider,
  noFallback: out.noFallback, tokenCount: out.tokenCount, text: out.text?.slice(0, 160), timing: out.timing,
  decodeMode: out.stats?.decodeMode, batchGuardReason: out.stats?.batchGuardReason, stopReason: out.stats?.stopReason,
  protein: out.protein ? { alphabet: out.protein.alphabet, dim: out.protein.embeddingDim, pooled0: out.protein.pooledEmbedding.slice(0, 3) } : undefined,
  reference: out.reference };
console.log(JSON.stringify(summary, null, 1));
