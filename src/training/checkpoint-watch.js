import { readdir, readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';

import { writeJsonArtifact } from './operator-artifacts.js';

async function listCheckpointMarkers(checkpointsDir) {
  const entries = await readdir(checkpointsDir, { withFileTypes: true });
  const markers = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const markerPath = join(checkpointsDir, entry.name, 'checkpoint.complete.json');
    try {
      await readFile(markerPath, 'utf8');
      markers.push(markerPath);
    } catch (error) {
      if (error?.code !== 'ENOENT') {
        throw error;
      }
    }
  }
  return markers.sort((left, right) => left.localeCompare(right));
}

async function readProcessedManifest(manifestPath) {
  try {
    const raw = await readFile(manifestPath, 'utf8');
    const parsed = JSON.parse(raw);
    const processed = Array.isArray(parsed?.processedCheckpointMarkers)
      ? parsed.processedCheckpointMarkers.filter((entry) => typeof entry === 'string')
      : [];
    return new Set(processed);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return new Set();
    }
    throw error;
  }
}

export async function watchFinalizedCheckpoints(options) {
  const checkpointsDir = resolve(String(options.checkpointsDir));
  const manifestPath = resolve(String(options.manifestPath));
  const pollIntervalMs = Number.isFinite(options.pollIntervalMs)
    ? Math.max(100, Math.floor(options.pollIntervalMs))
    : 2000;
  const stopWhenIdle = options.stopWhenIdle === true;
  const onCheckpoint = typeof options.onCheckpoint === 'function'
    ? options.onCheckpoint
    : null;
  if (!onCheckpoint) {
    throw new Error('watchFinalizedCheckpoints requires onCheckpoint(markerPath).');
  }

  const processed = await readProcessedManifest(manifestPath);
  let idlePolls = 0;
  for (;;) {
    const markers = await listCheckpointMarkers(checkpointsDir);
    let sawNewMarker = false;
    for (const markerPath of markers) {
      if (processed.has(markerPath)) continue;
      sawNewMarker = true;
      await onCheckpoint(markerPath);
      processed.add(markerPath);
      await writeJsonArtifact(manifestPath, {
        artifactType: 'training_checkpoint_watch_manifest',
        schemaVersion: 1,
        generatedAt: new Date().toISOString(),
        processedCheckpointMarkers: [...processed].sort((left, right) => left.localeCompare(right)),
      });
    }
    if (!sawNewMarker) {
      idlePolls += 1;
      if (stopWhenIdle && idlePolls > 0) {
        return {
          ok: true,
          processedCount: processed.size,
          manifestPath,
        };
      }
    } else {
      idlePolls = 0;
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, pollIntervalMs));
  }
}
