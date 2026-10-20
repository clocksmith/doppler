#!/usr/bin/env node
// Bounded, file-transport qualification driver. Applications own scheduling and
// transport; Doppler owns gradient production, round validation, and the update.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { bootstrapNativeTrainingHost, releaseNativeTrainingHost,
  loadNativeQwenTrainingPipeline, createNativeQwenLoRATrainer } from '../src/training.js';
import { getKernelCapabilities } from '../src/gpu/device.js';
import { loadLoRAFromManifest } from '../src/experimental/adapters/lora-loader.js';
import { computeCanonicalSha256 } from '../src/formats/canonical-hash.js';

if (process.argv.length !== 4 || process.argv[2] !== '--config') {
  throw new Error('Usage: node tools/qualify-community-lora.js --config <job.json>');
}
const config = JSON.parse(await readFile(resolve(process.argv[3]), 'utf8'));
if (!['prepare', 'contribute', 'apply', 'verify'].includes(config.action)) throw new Error('Unknown round action.');
if (!config.modelUrl || !config.outputPath || !config.trainer) throw new Error('Explicit modelUrl, outputPath and trainer are required.');
const source = {};
for (const path of ['src/experimental/training/native-qwen-lora.js', 'src/experimental/training/lora-training-round.js']) {
  source[path] = createHash('sha256').update(await readFile(new URL(`../${path}`, import.meta.url))).digest('hex');
}
await bootstrapNativeTrainingHost();
let loaded, trainer;
try {
  loaded = await loadNativeQwenTrainingPipeline(config.modelUrl);
  trainer = createNativeQwenLoRATrainer({ ...config.trainer, pipeline: loaded.pipeline,
    baseModelId: loaded.manifest.modelId, layerIdx: loaded.pipeline.modelConfig.numLayers - 1, module: 'down_proj' });
  let result;
  if (config.action === 'verify') {
    const manifest = JSON.parse(await readFile(resolve(config.exportDir, 'adapter.json'), 'utf8'));
    const weightBytes = await readFile(resolve(config.exportDir, manifest.weightsPath));
    const bytes = weightBytes.buffer.slice(weightBytes.byteOffset, weightBytes.byteOffset + weightBytes.byteLength);
    if (manifest.metadata?.baseManifestHash !== computeCanonicalSha256(loaded.manifest)) throw new Error('Export base artifact mismatch.');
    const adapter = await loadLoRAFromManifest(manifest, { readFile: async () => bytes });
    const corrupted = bytes.slice(0); new Uint8Array(corrupted)[corrupted.byteLength - 1] ^= 1;
    let corruptionRejected = false;
    try { await loadLoRAFromManifest(manifest, { readFile: async () => corrupted }); } catch { corruptionRejected = true; }
    if (!corruptionRejected) throw new Error('Corrupt adapter was accepted.');
    const observe = async (selection) => {
      loaded.pipeline.reset(); loaded.pipeline.setLoRAAdapter(selection);
      const observed = await loaded.pipeline.prefillWithLogits(config.inputIds, {
        temperature: 0, topK: 1, topP: 1, useChatTemplate: false,
      });
      const logits = observed.logits;
      if (!logits || !logits.every(Number.isFinite)) throw new Error('Missing finite reference logits.');
      return logits;
    };
    const base = await observe(null), adapted = await observe(adapter), restored = await observe(null);
    let adapterDelta = 0, restoredDelta = 0;
    for (let i = 0; i < base.length; i++) {
      adapterDelta = Math.max(adapterDelta, Math.abs(adapted[i] - base[i]));
      restoredDelta = Math.max(restoredDelta, Math.abs(restored[i] - base[i]));
    }
    if (!(adapterDelta > 0) || restoredDelta !== 0) throw new Error('Adapter effect or base restoration failed.');
    result = { adapterId: adapter.id, version: adapter.version, corruptionRejected,
      adapterDelta, restoredDelta, finiteLogits: base.length, qualityQualified: false };
  } else if (config.action === 'prepare') {
    result = await trainer.createTrainingRound({ jobId: config.jobId, roundId: config.roundId, assignments: config.assignments });
  } else {
    const saved = JSON.parse(await readFile(resolve(config.roundPath), 'utf8'));
    const round = saved.result;
    if (config.action === 'contribute') {
      result = await trainer.computeGradientContribution(round, config.participantId, config.sample);
    } else {
      trainer.restoreCheckpoint(round.checkpoint);
      const contributions = await Promise.all(config.contributionPaths.map(async (path) => (
        JSON.parse(await readFile(resolve(path), 'utf8')).result
      )));
      result = await trainer.applyGradientContributions(round, contributions);
      const adapter = await trainer.exportAdapter({ ...config.adapter,
        metadata: { trainingRoundHash: round.roundHash, parentCheckpointHash: round.checkpointHash,
          participants: result.contributors, supervisedTokenCount: result.supervisedTokenCount } });
      const exportDir = resolve(config.exportDir);
      await mkdir(exportDir, { recursive: true });
      await writeFile(resolve(exportDir, config.adapter.weightsPath), new Uint8Array(adapter.weights));
      await writeFile(resolve(exportDir, 'adapter.json'), adapter.json);
      await writeFile(resolve(exportDir, 'checkpoint.json'), JSON.stringify(await trainer.createCheckpoint()));
      result = { ...result, adapterManifest: adapter.manifest, weightsSha256: adapter.weightsSha256 };
    }
  }
  await mkdir(dirname(resolve(config.outputPath)), { recursive: true });
  await writeFile(resolve(config.outputPath), JSON.stringify({ schema: 'doppler.community-lora-qualification/v1',
    action: config.action, source, device: getKernelCapabilities().adapterInfo,
    modelId: loaded.manifest.modelId, baseManifestHash: computeCanonicalSha256(loaded.manifest), result }, null, 2) + '\n');
  console.log(`Wrote ${config.outputPath}`);
} finally {
  trainer?.dispose();
  await loaded?.pipeline?.unload();
  await releaseNativeTrainingHost();
}
