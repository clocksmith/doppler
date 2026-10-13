import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { getKernelConfigs } from '../../../src/config/kernel-registry-contract.js';
const read = async name => JSON.parse(await fs.readFile(new URL(name, import.meta.url), 'utf8'));
const oldGraph = await read('../execution.json'), newGraph = await read('execution.json');
const observations = await read('observed-pipeline-inputs.json');
const configs = Object.values(getKernelConfigs()).flatMap(Object.values);
const reachable = graph => new Set([...graph.mechanismKernels, ...graph.preLayer.map(s=>s[1]),
 ...graph.decode.map(s=>s[1]), ...graph.prefill.map(s=>s[1]), ...graph.postLayer.map(s=>s[1])]
 .map(id=>graph.kernels[id]).map(k=>`${k.kernel}#${k.entry}`));
const oldClosure=reachable(oldGraph), newClosure=reachable(newGraph), observed=new Map();
for (const row of observations.rows) for (const pipeline of row.pipelines) {
 const matches=configs.filter(c=>pipeline.label===`${c.operation}_${c.variant}_pipeline`
  || pipeline.label.startsWith(`${c.operation}_${c.variant}_pipeline_`));
 assert.equal(matches.length,1,pipeline.label); const c=matches[0];
 assert.equal(c.entryPoint,pipeline.entryPoint); const key=`${c.shaderFile}#${c.entryPoint}`;
 assert(newClosure.has(key),`Missing actual observed pipeline ${key}`);
 observed.set(key,{file:c.shaderFile,entry:c.entryPoint,operation:c.operation,variant:c.variant});
}
const missing=[...observed.keys()].filter(key=>!oldClosure.has(key)).sort();
assert.deepEqual(missing,['gather_f16_vec4.wgsl#gather_vec4','logit_finalize_f32.wgsl#main']);
const artifact = {scope:observations.scope,missingFromPrior:missing,observedPipelines:[...observed.values()],
 declaredFallback:'gather_split8_f16_vec4_f32_out.wgsl F16→F32 is additionally declared for existing split embedding dispatch; scalar gather remains reachable. Neither fallback is claimed observed in these receipts.',
 inputSha256:createHash('sha256').update(await fs.readFile(new URL('observed-pipeline-inputs.json',import.meta.url))).digest('hex')};
await fs.writeFile(new URL('pipeline-audit.json',import.meta.url),JSON.stringify(artifact,null,2)+'\n');
console.log(JSON.stringify({observed:observed.size,missingFromPrior:missing,newClosureCoversObserved:true}));
