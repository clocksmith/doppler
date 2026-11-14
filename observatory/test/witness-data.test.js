import test from 'node:test';
import assert from 'node:assert/strict';
import { numbers, activityOf, tokenRow, attentionWitness, gearboxProducts, memoryLandscape } from '../public/views/witness-data.js';
const frame=(role,values,extra={})=>({role,values,status:'captured',representation:'raw',layer:0,tokenStart:0,tokenIds:[1,2],fullShape:[2,2],atMs:1,...extra});
const query=frame('attention-query',[1,0,1,0],{atMs:2,metadata:{heads:1,kvHeads:1,headDim:2,kvLength:2,scale:1,softcap:0,slidingWindow:null,causal:true}});
const activity=()=>({frames:[frame('kv-key',[1,0,0,1]),query],weights:[]});
test('packed little-endian floats retain exact values and bounds',()=>{const b=Buffer.alloc(8);b.writeFloatLE(1.25,0);b.writeFloatLE(-2,4);assert.deepEqual(numbers({encoding:'base64-le',dtype:'f32',length:2,data:b.toString('base64')}),[1.25,-2]);assert.throws(()=>numbers({encoding:'base64-le',dtype:'f32',length:2,data:''}));});
test('activity identity must match run and absent is distinct',()=>{assert.equal(activityOf({}),null);assert.throws(()=>activityOf({model:{manifestHash:'a'},activityTrace:{identity:{manifestHash:'b'},activity:activity()}}));});
test('truncated token rows remain missing',()=>{assert.deepEqual(tokenRow(frame('kv-key',[1,0]),1),[]);});
test('attention reconstructs a complete normalized distribution and causal mask',()=>{const a=activity();assert.deepEqual(attentionWitness(a,0,0).probabilities,[1,0]);const p=attentionWitness(a,1,0).probabilities;assert.ok(Math.abs(p[0]-Math.E/(Math.E+1))<1e-12);assert.ok(Math.abs(p[0]+p[1]-1)<1e-12);});
test('missing unmasked key refuses partial softmax, masked missing is safe',()=>{const a=activity();a.frames[0].values=[1,0];assert.equal(attentionWitness(a,1,0),null);assert.deepEqual(attentionWitness(a,0,0).probabilities,[1,0]);});
test('window and grouped-query mapping use recorded metadata',()=>{const a=activity();a.frames[1]={...query,fullShape:[2,4],values:[1,0,0,1,1,0,0,1],metadata:{...query.metadata,heads:2,slidingWindow:1}};assert.deepEqual(attentionWitness(a,1,0,1).probabilities,[0,1]);assert.equal(attentionWitness(a,1,0,2),null);});
test('failed latest captures do not reuse older successful values',()=>{const a=activity();a.frames.push(frame('kv-key',null,{status:'dropped',atMs:2}));assert.equal(attentionWitness(a,1,0),null);});
test('gearbox is signed weight times matched input, refuses adapters and unrelated weights',()=>{const a={frames:[frame('ffn-down-input',[2,-3,4,5])],weights:[]};const w={layer:0,role:'down',layout:'row',status:'captured',shape:[2,2],rawOffset:0,raw:[1,2,-1,3],columnEnergy:[2,13]};assert.deepEqual(gearboxProducts(a,w,0).products.map(p=>p.product),[2,-6,-2,-9]);assert.equal(gearboxProducts({...a,identity:{adapters:['a']}},w,0),null);assert.equal(gearboxProducts(a,{...w,columnEnergy:[]},0),null);assert.equal(gearboxProducts(a,{...w,role:'up'},0),null);assert.equal(gearboxProducts(a,{...w,layout:'column'},0),null);});
test('memory keeps missing positions distinct from captured zero',()=>{const a={frames:[frame('kv-key',[0,0])],weights:[]};const rows=memoryLandscape(a,1,0);assert.equal(rows[0].value,0);assert.equal(rows[1].value,null);});
test('recurrent prefill state preserves chunk boundary, not per-token claim',()=>{const rows=memoryLandscape({frames:[frame('recurrent-memory',[3,4])],weights:[]},0,0);assert.equal(rows[0].count,2);assert.match(rows[0].boundary,/positions 0–1/);});
test('softcap is applied before stable softmax',()=>{const a=activity();a.frames[1]={...query,metadata:{...query.metadata,softcap:.5}};const expected=Math.exp(.5*Math.tanh(2));assert.ok(Math.abs(attentionWitness(a,1,0).probabilities[0]-expected/(expected+1))<1e-12);});
test('each query head uses its grouped KV head',()=>{const a=activity();a.frames[1]={...query,fullShape:[2,4],values:[1,0,0,1,1,0,0,1],metadata:{...query.metadata,heads:2}};assert.ok(attentionWitness(a,1,0,0).probabilities[0]>.5);assert.ok(attentionWitness(a,1,0,1).probabilities[0]<.5);});
test('oversized attention work is rejected before reconstruction',()=>{const a=activity();a.frames[1]={...query,metadata:{...query.metadata,kvLength:5000}};assert.throws(()=>attentionWitness(a,1,0),/budget/);});

test('nearestRecorded searches outward and prefers the earlier position on a tie', async () => {
  const { nearestRecorded } = await import('../public/views/witness-data.js');
  assert.equal(nearestRecorded(5, 10, (p) => p === 3 || p === 7), 3);
  assert.equal(nearestRecorded(5, 10, (p) => p === 8), 8);
  assert.equal(nearestRecorded(0, 3, () => false), null);
});
