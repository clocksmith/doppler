#!/usr/bin/env node
import fs from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import sharp from 'sharp';
import previews from '../demo/ui/activation-grid/previews.json' with { type: 'json' };
import { validateRecording, colorGrid, initialColorScale } from '../demo/ui/activation-grid/data.js';
import { activityView, playbackPoints } from '../demo/ui/activation-grid/views.js';

// The first-load images are exact pixels from the public recorded run, not art.
const root = new URL('../demo/assets/', import.meta.url);
const metadata = JSON.parse(await fs.readFile(new URL('activity-example.json', root), 'utf8'));
const bytes = await fs.readFile(new URL('activity-example.json.gz', root));
if (createHash('sha256').update(bytes).digest('hex') !== metadata.sha256) throw new Error('Example hash differs.');
const recording = validateRecording(JSON.parse(gunzipSync(bytes)));
const point = playbackPoints(recording, 'tokens').find(p => p.position === metadata.promptPosition);
if (!point) throw new Error('Example position is missing.');
const receipt = { sourceSha256: metadata.sha256, model: metadata.model.logicalModelId, position: point.position, scaleRule: "fixed 99th-percentile magnitude; probabilities use 0–1; one scalar per opaque 24-bit heatmap pixel", previews: [] };
const check = process.argv.includes('--check');
for (const preview of previews) {
  const view = activityView(recording, { mode: preview.mode, position: point.position, atMs: point.atMs, layer: -1, weightId: '', detail: 'overview', page: 0, baseline: null });
  const scale = view.range === 'unit' || view.range === 'residency' ? 1 : initialColorScale(view.values, recording.display.scale);
  const { side, rgba } = colorGrid(view.values, scale, view.range);
  if (!side || !view.values.some(v => v !== undefined)) throw new Error(`Empty preview: ${preview.mode}`);
  const png = await sharp(Buffer.from(rgba), { raw: { width: side, height: side, channels: 4 } }).removeAlpha().png({ compressionLevel: 9, adaptiveFiltering: false }).toBuffer();
  const file = new URL(`activity-preview-${preview.mode}.png`, root);
  if (check) { if (!png.equals(await fs.readFile(file))) throw new Error(`Stale preview: ${preview.mode}`); }
  else await fs.writeFile(file, png);
  receipt.previews.push({ mode: preview.mode, side, scale, range: view.range, representation: view.representation, underlyingValues: view.underlyingValues, sha256: createHash('sha256').update(png).digest('hex') });
}
// Shipped in the demo shell, which has a byte budget: one preview per line.
const { previews: items, ...head } = receipt;
const manifest = `{\n${Object.entries(head).map(([key, value]) => `  ${JSON.stringify(key)}: ${JSON.stringify(value)},\n`).join('')}  "previews": [\n${items.map(item => `    ${JSON.stringify(item)}`).join(',\n')}\n  ]\n}\n`;
const target = new URL('activity-previews.json', root);
if (check) { if (manifest !== await fs.readFile(target, 'utf8')) throw new Error('Stale preview manifest.'); }
else await fs.writeFile(target, manifest);
console.log(`${check ? 'Verified' : 'Generated'} ${previews.length} recorded activity previews.`);
