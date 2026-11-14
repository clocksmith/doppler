import test from 'node:test';
import assert from 'node:assert/strict';
import { renderGearbox } from '../public/views/gearbox.js';
import { renderAttention } from '../public/views/attention.js';
import { renderMemory } from '../public/views/memory.js';
class Node {
  constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.attrs={};this.events={};}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  setAttribute(k,v){this.attrs[k]=String(v);}
  addEventListener(k,v){this.events[k]=v;}
}
function descendants(n,tag){return n.children.filter(c=>c instanceof Node).flatMap(c=>[...(c.tag===tag?[c]:[]),...descendants(c,tag)]);}
function doc(){const roots=Object.fromEntries(['gearbox','attention','memory'].map(k=>[k,new Node('div')]));return {roots,getElementById:k=>roots[k],createElement:t=>new Node(t),createElementNS:(_,t)=>new Node(t)};}
const frame=(role,values,extra={})=>({role,values,status:'captured',representation:'raw',layer:0,tokenStart:0,tokenIds:[1,2],fullShape:[2,2],atMs:1,...extra});
function context(){return {run:{id:'test',model:{manifestHash:'same'},activityTrace:{identity:{manifestHash:'same'},activity:{frames:[frame('kv-key',[1,0,0,1]),frame('kv-value',[1,2,3,4]),frame('ffn-down-input',[1,2,3,4]),frame('attention-query',[1,0,0,1],{atMs:2,metadata:{heads:1,kvHeads:1,headDim:2,kvLength:2,scale:1,softcap:0,slidingWindow:null,causal:true}})],weights:[{id:'down',role:'down',layout:'row',status:'captured',layer:0,shape:[2,2],raw:[1,-2,3,4],rawOffset:0,columnEnergy:[10,20]}]}}},selection:{get:()=>({position:1,layer:0}),select(){}}};}
test('three views render honest gaps without activity',()=>{globalThis.document=doc();const c={run:{},selection:{get:()=>({position:1,layer:0})}};for(const [id,render] of [['gearbox',renderGearbox],['attention',renderAttention],['memory',renderMemory]]){render(c);assert.ok(document.roots[id].children.length);}delete globalThis.document;});
test('three views render captured values and their controls',()=>{globalThis.document=doc();const c=context();renderGearbox(c);renderAttention(c);renderMemory(c);assert.equal(descendants(document.roots.gearbox,'rect').length,4);assert.equal(descendants(document.roots.attention,'path').length,2);assert.equal(descendants(document.roots.memory,'button').length,4);const head=descendants(document.roots.attention,'select')[1];head.events.change({target:{value:'0'}});assert.equal(descendants(document.roots.attention,'path').length,2);let chosen;c.selection.select=value=>{chosen=value;};descendants(document.roots.memory,'button')[0].events.click();assert.deepEqual(chosen,{position:0,layer:0});delete globalThis.document;});
test('malformed payload becomes a view gap rather than page exception',()=>{globalThis.document=doc();const c=context();c.run.activityTrace.activity.frames[0].values={encoding:'broken'};assert.doesNotThrow(()=>renderAttention(c));assert.equal(descendants(document.roots.attention,'path').length,0);delete globalThis.document;});
test('gearbox keeps the tensor picker when a selected tensor lacks matching input', () => {
  globalThis.document = doc();
  const c = context();
  c.run.id = 'picker-recovery';
  c.run.activityTrace.activity.weights.push({ ...c.run.activityTrace.activity.weights[0], id: 'missing-input', shape: [2, 3] });
  renderGearbox(c);
  descendants(document.roots.gearbox, 'select')[0].events.change({ target: { value: 'missing-input' } });
  assert.equal(descendants(document.roots.gearbox, 'rect').length, 0);
  const picker = descendants(document.roots.gearbox, 'select')[0];
  assert.ok(picker, 'the missing-data view still allows switching tensors');
  picker.events.change({ target: { value: 'down' } });
  assert.equal(descendants(document.roots.gearbox, 'rect').length, 4);
  delete globalThis.document;
});
