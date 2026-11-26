// CPU-only probe of values the installed dopplerx emits through PUBLIC subpaths
// (paths resolved through the package's own exports map; no GPU device is requested).
import { readFileSync } from 'node:fs';
const base = process.argv[2] + '/node_modules/dopplerx/';
const pkg = JSON.parse(readFileSync(base + 'package.json', 'utf8'));
const sub = (s) => import('file://' + base + pkg.exports[s].import.replace(/^\.\//, ''));
const out = { version: pkg.version, gpuPresent: typeof globalThis.navigator?.gpu !== 'undefined' };
const mm = await sub('./client/model-manager');
out['client/model-manager activateLoRAFromTrainingOutput(null) with no model'] = await mm.activateLoRAFromTrainingOutput(null);
const tr = await sub('./training');
out['training TRAINING_BACKENDS'] = tr.TRAINING_BACKENDS;
try { tr.assertTrainingBackend({}, 'bogus'); } catch (e) { out['training assertTrainingBackend error'] = e.message; }
console.log(JSON.stringify(out, null, 2));
