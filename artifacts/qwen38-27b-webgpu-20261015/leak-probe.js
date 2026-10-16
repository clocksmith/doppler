import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { doppler } from '../../src/index.js';
import { getBufferPool } from '../../src/memory/buffer-pool.js';
import { getDevice } from '../../src/gpu/device.js';

const [modelDir, outputDir] = process.argv.slice(2);
const policy = JSON.parse(await fs.readFile(new URL('./policy-recorded.json', import.meta.url), 'utf8'));
const record = { policy, observations: [] };
let model;
let device;
function observe(phase) {
  const pool = getBufferPool(device);
  record.observations.push({ phase, stats: pool.getStats(), labels: pool.getLabelStats()
    .filter(row => !row.label.startsWith('raw_')).sort((a, b) => b.bytes - a.bytes) });
}
try {
  model = await doppler.load({ url: pathToFileURL(path.resolve(modelDir) + path.sep).href }, {
    runtimeConfig: policy.runtimeConfig,
  });
  device = getDevice();
  observe('loaded');
  record.result = await model.chatText([{ role: 'user', content: policy.prompt }], policy.options);
  observe('after-one-token');
} catch (error) {
  record.error = { message: error.message, stack: error.stack };
} finally {
  await model?.unload();
  if (device) observe('unloaded');
  await fs.writeFile(path.join(outputDir, 'result.json'), JSON.stringify(record, null, 2));
  device?.destroy();
}
process.exit(record.error ? 1 : 0);
