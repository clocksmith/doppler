// Loads a model through the PUBLIC entry points with a manifest object (as Reploid's
// doppler-runtime does: { manifest, baseUrl }), on the real GPU. Reports load success or the error.
// usage: node public-manifest-object-probe.mjs <install-dir> <model-dir>
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const [installDir, modelDir] = process.argv.slice(2);
const root = path.join(installDir, 'node_modules/dopplerx');
const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
const sub = (s) => { const e = pkg.exports[s]; const f = typeof e === 'string' ? e : (e.node ?? e.import ?? e.default); return import(pathToFileURL(path.join(root, typeof f === 'string' ? f : f.import ?? f.default)).href); };
const manifest = JSON.parse(fs.readFileSync(path.join(modelDir, 'manifest.json'), 'utf8'));
const { bootstrapNodeWebGPU } = await import(pathToFileURL(path.join(root, 'src/tooling/node-webgpu.js')).href);
await bootstrapNodeWebGPU();
const out = { version: pkg.version };
const { dr } = await sub('./compat');
try {
  const model = await dr.load({ manifest, baseUrl: pathToFileURL(modelDir).href + '/' });
  out.compatLoadManifestObject = { ok: true };
  await model.unload?.();
} catch (e) { out.compatLoadManifestObject = { ok: false, message: String(e.message).slice(0, 220) }; }
console.log(JSON.stringify(out, null, 1));
process.exit(0);
