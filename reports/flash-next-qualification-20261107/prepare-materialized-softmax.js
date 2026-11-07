import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';

// Retained isolated snapshot only. Preserve Q/K dot order and Q4K/model state;
// materialize source-style normalized F32 probabilities before the V products.
const file = new URL('../../src/gpu/kernels/attention_indexed.wgsl', import.meta.url);
const raw = await fs.readFile(file);
const hash = value => createHash('sha256').update(value).digest('hex');
assert.equal(hash(raw),'49a68e1b3a635be70d123167a40c9452c7e7a228978931552be9644d1abd703d');
let text = raw.toString('utf8');
const replace = (a,b) => { assert.equal(text.split(a).length-1,1); text=text.replace(a,b); };
replace('override WORKGROUP_SIZE: u32 = 256u;', 'override WORKGROUP_SIZE: u32 = 256u;\noverride MAX_SELECTED: u32;');
replace('var<workgroup> shared_p: array<f32, WORKGROUP_SIZE>;', 'var<workgroup> probabilities: array<f32, MAX_SELECTED>;');
replace('var<workgroup> shared_token: array<u32, WORKGROUP_SIZE>;\n','');
const first = text.indexOf('    var running_max = NEG_LARGE;');
const last = text.indexOf('    let out_base =',first);
assert(first>0 && last>first);
text=text.slice(0,first)+`    var local_max = NEG_LARGE;
    for (var index = local; index < count; index += WORKGROUP_SIZE) {
        let cached = select(index, selection[token * u.max_selected + index], listed);
        let k_base = cached * row_stride + kv_head * head_dim;
        var dot = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            dot = fma(shared_q[d], K[k_base + d], dot);
        }
        let score = dot * u.scale;
        probabilities[index] = score;
        local_max = max(local_max, score);
    }
    let maximum = tree_max(local, local_max);
    var local_sum = 0.0;
    for (var index = local; index < count; index += WORKGROUP_SIZE) {
        let p = exp_refined(probabilities[index] - maximum);
        probabilities[index] = p;
        local_sum += p;
    }
    let denominator = tree_sum(local, local_sum);
    for (var index = local; index < count; index += WORKGROUP_SIZE) {
        probabilities[index] = probabilities[index] / denominator;
    }
    workgroupBarrier();
    let chunks = (count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    for (var chunk = 0u; chunk < chunks; chunk++) {
        var sum0 = 0.0;
        var sum1 = 0.0;
        let end = min(count, (chunk + 1u) * WORKGROUP_SIZE);
        for (var index = chunk * WORKGROUP_SIZE; index < end; index++) {
            let cached = select(index, selection[token * u.max_selected + index], listed);
            let v_base = cached * row_stride + kv_head * head_dim;
            if (dim0 < head_dim) { sum0 = fma(probabilities[index], V[v_base + dim0], sum0); }
            if (dim1 < head_dim) { sum1 = fma(probabilities[index], V[v_base + dim1], sum1); }
        }
        acc0 += sum0;
        acc1 += sum1;
    }

`+text.slice(last);
replace('select(0.0, acc0 / running_sum, running_sum > 0.0)', 'select(0.0, acc0, denominator > 0.0)');
replace('select(0.0, acc1 / running_sum, running_sum > 0.0)', 'select(0.0, acc1, denominator > 0.0)');
text=text.replace('Softmax is computed online in f32 over chunks;', 'Normalized F32 probabilities are materialized before V accumulation;');
await fs.writeFile(file,text);
const wrapper = new URL('../../src/gpu/kernels/qsa-indexer.js',import.meta.url);
let js=await fs.readFile(wrapper,'utf8');
const anchor='[numHeads, numTokens, 1], { WORKGROUP_SIZE: WIDE_WORKGROUP });';
assert.equal(js.split(anchor).length-1,1);
js=js.replace(anchor,`[numHeads, numTokens, 1], { WORKGROUP_SIZE: WIDE_WORKGROUP,
        ...(storage.selectionCapacityOverride ? { [storage.selectionCapacityOverride]: selection.maxSelected } : {}) });`);
await fs.writeFile(wrapper,js);
const rulesFile=new URL('../../src/rules/kernels/attention.rules.json',import.meta.url);
const rules=JSON.parse(await fs.readFile(rulesFile));
const rule=rules.indexedStorage.find(row=>row.match.kvDtype==='f32');
assert(rule);
rule.value.selectionCapacityOverride='MAX_SELECTED';
await fs.writeFile(rulesFile,JSON.stringify(rules,null,2)+'\n');
const registryFile=new URL('../../src/config/kernels/registry.json',import.meta.url);
const registry=JSON.parse(await fs.readFile(registryFile));
// Specialization is required from the actual selection geometry, never a
// registry default. The wrapper passes MAX_SELECTED before pipeline creation.
await fs.writeFile(registryFile,JSON.stringify(registry,null,1)+'\n');
console.log(JSON.stringify({scope:'Isolated normalized-probability materialization candidate; original Q/K dot order preserved',
  baselineSha256:hash(raw),candidateSha256:hash(text),wrapperSha256:hash(js),generatorSha256:hash(await fs.readFile(new URL(import.meta.url))),qualified:false}));
