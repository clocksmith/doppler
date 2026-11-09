import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { validateRecording } from '../../demo/ui/activation-grid/data.js';
import { encodeRecording } from '../../demo/ui/activation-grid/recording-codec.js';
import { modelPathView } from '../../demo/ui/activation-grid/paths.js';
import { activityModes, activityView } from '../../demo/ui/activation-grid/views.js';
import { SHELL_MANIFEST_DIGEST } from '../../demo/generated-shell-manifest.js';

const source = process.argv[2];
assert(source, 'Supply the physical inspection receipt to package.');
const receipt = JSON.parse(await fs.readFile(source, 'utf8'));
const prompt = receipt.prompt ?? process.argv[3];
assert(typeof prompt === 'string' && prompt.trim(), 'Supply the exact recorded prompt as the second argument when the receipt omits its text.');
const recording = validateRecording({ schema: 'doppler.activation-recording/v1',
  demoBuild: SHELL_MANIFEST_DIGEST,
  prompt, display: { layout: 'scalar-square/v2', palette: 'opaque-heatmap/v1', scale: 1 },
  trace: receipt.activationTrace, fingerprint: receipt.fingerprint,
  inputTokenIds: receipt.inputTokenIds, generatedTokenIds: receipt.generatedTokenIds, outputText: receipt.outputText });
const promptPosition = recording.inputTokenIds.length - 1;
const modes = activityModes(recording).map(mode => {
  const view = mode.id === 'model-paths' ? {
    values: modelPathView(recording, Infinity, promptPosition, -1).outputs,
    underlyingValues: recording.trace.geometry.layers * recording.trace.geometry.width,
    representation: 'raw', quantity: 'Layer outputs in the computation diagram',
    description: 'Rows are layers, columns are hidden dimensions; captured operator structure accompanies the map.',
  } : activityView(recording, { mode: mode.id, atMs: Infinity, position: promptPosition,
    layer: -1, weightId: '', detail: 'overview', page: 0 });
  const present = view.values.filter(Number.isFinite).length;
  assert(mode.reason || present > 0, `Enabled ${mode.id} lacks a real example.`);
  return { ...mode, displayedScalars: view.values.length, present, side: Math.ceil(Math.sqrt(view.values.length)),
    underlyingValues: view.underlyingValues, representation: view.representation, quantity: view.quantity, description: view.description };
});
const packed = JSON.stringify(encodeRecording(recording));
assert.deepEqual(validateRecording(JSON.parse(packed)), recording, 'Saved example must retain every numerical value.');
const compressed = gzipSync(packed, { level: 9 });
await fs.mkdir('demo/assets', { recursive: true });
await fs.writeFile('demo/assets/activity-example.json.gz', compressed);
const summary = { schema: 'doppler.activity-example/v1', sha256: createHash('sha256').update(compressed).digest('hex'),
  compressedBytes: compressed.length, decodedBytes: Buffer.byteLength(packed), model: recording.trace.identity,
  promptPosition, prompt, outputText: receipt.outputText, modes };
await fs.writeFile('demo/assets/activity-example.json', JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({ compressedBytes: compressed.length, decodedBytes: summary.decodedBytes,
  enabled: modes.filter(m => !m.reason).length, unavailable: modes.filter(m => m.reason).map(m => m.id) }));
