import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { validateRecording } from '../../demo/ui/activation-grid/data.js';
import { encodeRecording } from '../../demo/ui/activation-grid/recording-codec.js';
import { activityModes, activityView } from '../../demo/ui/activation-grid/views.js';
import { SHELL_MANIFEST_DIGEST } from '../../demo/generated-shell-manifest.js';

const source = process.argv[2];
assert(source, 'Supply the physical inspection receipt to package.');
const receipt = JSON.parse(await fs.readFile(source, 'utf8'));
const recording = validateRecording({ schema: 'doppler.activation-recording/v1',
  demoBuild: SHELL_MANIFEST_DIGEST,
  display: { layout: 'layer-major-rgb/v1', palette: 'signed-tanh-rgb/v1', scale: 1 },
  trace: receipt.activationTrace, fingerprint: receipt.fingerprint,
  inputTokenIds: receipt.inputTokenIds, generatedTokenIds: receipt.generatedTokenIds, outputText: receipt.outputText });
const promptPosition = recording.inputTokenIds.length - 1;
const modes = activityModes(recording).map(mode => {
  const view = activityView(recording, { mode: mode.id, atMs: Infinity, position: promptPosition,
    layer: -1, weightId: '', detail: 'overview', page: 0 });
  const present = view.values.filter(Number.isFinite).length;
  assert(mode.reason || present > 0, `Enabled ${mode.id} lacks a real example.`);
  return { ...mode, displayedScalars: view.values.length, present, side: Math.ceil(Math.sqrt(view.values.length / 3)),
    underlyingValues: view.underlyingValues, representation: view.representation, quantity: view.quantity, description: view.description };
});
const packed = JSON.stringify(encodeRecording(recording));
assert.deepEqual(validateRecording(JSON.parse(packed)), recording, 'Saved example must retain every numerical value.');
const compressed = gzipSync(packed, { level: 9 });
await fs.mkdir('demo/assets', { recursive: true });
await fs.writeFile('demo/assets/activity-example.json.gz', compressed);
const summary = { schema: 'doppler.activity-example/v1', sha256: createHash('sha256').update(compressed).digest('hex'),
  compressedBytes: compressed.length, decodedBytes: Buffer.byteLength(packed), model: recording.trace.identity,
  promptPosition, outputText: receipt.outputText, modes };
await fs.writeFile('demo/assets/activity-example.json', JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({ compressedBytes: compressed.length, decodedBytes: summary.decodedBytes,
  enabled: modes.filter(m => !m.reason).length, unavailable: modes.filter(m => m.reason).map(m => m.id) }));
