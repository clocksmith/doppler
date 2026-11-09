// Offline repository preparation. This does not execute, sign or approve a route.
import fs from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';
import { createOrderedLayerPartitionPlanFromManifest, hashLayerPartitionPlan } from '../src/config/layer-partition-plan.js';

const { values } = parseArgs({ options: {
  manifest: { type: 'string' }, config: { type: 'string' }, out: { type: 'string' },
}, strict: true, allowPositionals: false });
if (!values.manifest || !values.config || !values.out) {
  throw new Error('Usage: node tools/compile-layer-partition-plan.js --manifest manifest.json --config split.json --out plan.json');
}
const manifestBytes = await fs.readFile(values.manifest);
const manifest = JSON.parse(manifestBytes);
const request = JSON.parse(await fs.readFile(values.config, 'utf8'));
const plan = createOrderedLayerPartitionPlanFromManifest(manifest, request);
const identity = 'sha256:' + createHash('sha256').update(manifestBytes).digest('hex');
const result = { model: { id: manifest.modelId, identity }, plan, planId: hashLayerPartitionPlan(plan),
  qualification: null };
await fs.writeFile(values.out, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ modelId: manifest.modelId, planId: result.planId, stages: plan.stageCount,
  qualified: false, output: values.out }));
