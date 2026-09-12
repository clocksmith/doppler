import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import vm from 'node:vm';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { referenceExpansion, shaderData, stripWGSLComments, interpret, referencePixels } from './reference.mjs';
import { assemble } from '../compiler/assemble.mjs';
import assert from 'node:assert/strict';
const ROOT=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const read=p=>fs.readFileSync(path.join(ROOT,p),'utf8');
const sha=s=>crypto.createHash('sha256').update(s).digest('hex');
function load(source) {
  const context=vm.createContext({__QUINE_LIBRARY__:true,TextDecoder,TextEncoder,console});
  vm.runInContext(source,context,{timeout:5000});
  if(!context.Quine)throw Error('Quine API missing');
  return context.Quine;
}
const checks=[];
function check(name,fn){const start=performance.now();fn();checks.push({name,passed:true,elapsedMs:Math.round(performance.now()-start)});console.log('PASS',name);}
const filenames=['01-application','02-polyglot','03-multiquine','04-visual','05-compiler'];
const sources=Object.fromEntries(filenames.map(n=>[n,read('quines/'+n+'.js')]));
const apis=Object.fromEntries(filenames.map(n=>[n,load(sources[n])]));
for(const name of filenames) {
 check(name+' / JS reconstruction and emitted WGSL',()=>{
  assert.equal(apis[name].self(),sources[name]);
  assert.equal(apis[name].wgsl(),read('quines/'+name+'.wgsl'));
 });
 check(name+' / no source-reading mechanism',()=>{
  for(const bad of [/Function\.prototype\.toString/,/document\.currentScript/,/readFile/,/fetch\(/])assert.equal(bad.test(sources[name]),false,String(bad));
 });
}
check('complete application / 32 descendant rounds through independent shader reference',()=>{
 let current=sources['01-application'];
 for(let i=0;i<32;i++){const child=referenceExpansion(load(current).wgsl());assert.equal(child,current);current=child;}
});
check('polyglot / identical file bytes and 32 mixed-language rounds',()=>{
 let current=sources['02-polyglot'];
 assert.equal(current,read('quines/02-polyglot.wgsl'));
 const visible=stripWGSLComments(current);
 assert(!visible.includes('globalThis'));assert(visible.includes('@compute'));
 for(let i=0;i<32;i++) { const child=i%2?load(current).self():referenceExpansion(current);assert.equal(child,current);current=child; }
});
check('multiquine / all four transitions and 64 mixed transitions',()=>{
 const a=sources['03-multiquine'],b=read('quines/03-multiquine.wgsl');
 assert.equal(load(a).emit('js'),a);assert.equal(load(a).emit('wgsl'),b);
 assert.equal(referenceExpansion(b,'js'),a);assert.equal(referenceExpansion(b,'wgsl'),b);
 let current=a,lang='js';
 for(let i=0;i<64;i++) {
  const target=(i*7%5)<2?'js':'wgsl';
  current=lang==='js'?load(current).emit(target):referenceExpansion(current,target);
  assert.equal(current,target==='js'?a:b);lang=target;
 }
});
check('visual / source reconstruction and lossless reference pixel decoding',()=>{
 const api=apis['04-visual']; const text=referenceExpansion(api.wgsl());assert.equal(text,sources['04-visual']);
 const {rgba,width,height}=referencePixels(text,api.wgsl());
 assert.equal(api.decodeImage(rgba,width,height),text);
 const damaged=new Uint8ClampedArray(rgba);damaged[4*(9*width+15)+1]^=1;
 assert.throws(()=>api.decodeImage(damaged,width,height),/Damaged/);
 fs.writeFileSync(path.join(ROOT,'verification/visual-reference.rgba'),rgba);
 fs.writeFileSync(path.join(ROOT,'verification/visual-reference.json'),JSON.stringify({width,height,sourceSha256:sha(text),renderer:'independent CPU reference; not GPU evidence'},null,2)+'\n');
});
check('Sprout / human-readable assembler equals embedded compiler input',()=>{
 const p=JSON.parse(read('compiler/compiler-program.json'));
 assert.deepEqual(assemble(read('compiler/compiler.sprout')),p);
 assert.deepEqual(Array.from(apis['05-compiler'].program),p);
 assert.deepEqual(shaderData(read('quines/05-compiler.wgsl'),'P'),p);
});
check('Sprout / independent interpreter compiles its compiler to both exact backends',()=>{
 const p=JSON.parse(read('compiler/compiler-program.json'));
 assert.equal(interpret(p,p,0).text,sources['05-compiler']);
 assert.equal(interpret(p,p,1).text,read('quines/05-compiler.wgsl'));
});
check('Sprout / 16 alternating self-compilation rounds',()=>{
 let current=sources['05-compiler'];
 for(let i=0;i<16;i++) {
  const api=load(current), w=api.wgsl();
  assert.equal(w,read('quines/05-compiler.wgsl'));
  current=interpret(shaderData(w,'P'),undefined,0).text;
  assert.equal(current,sources['05-compiler']);
 }
});
for(const {name,expected} of JSON.parse(read('compiler/examples.json'))) {
 check('Sprout / compile and execute independent '+name+' example',()=>{
  const p=JSON.parse(read('compiler/'+name+'.json'));
  assert.deepEqual(assemble(read('compiler/'+name+'.sprout')),p);
  const compiler=apis['05-compiler'];
  const js=compiler.compile(p,'js'), wgsl=compiler.compile(p,'wgsl');
  assert.equal(js,read('compiler/'+name+'.js'));assert.equal(wgsl,read('compiler/'+name+'.wgsl'));
  assert.equal(load(js).execute(),expected);assert.equal(interpret(p).text,expected);
  assert.equal(interpret(Array.from(compiler.program),p,0).text,js);
  assert.equal(interpret(Array.from(compiler.program),p,1).text,wgsl);
 });
}
check('Sprout / rejects malformed headers, opcodes, and unclosed loops',()=>{
 const api=apis['05-compiler'];
 for(const p of [[0,0,0],[21331,0,4,99,0,0,0],[21331,0,4,14,0,0,0]])assert.throws(()=>api.compile(p));
});
check('standalone CLI / exact stdout for all five JS and WGSL sources',()=>{
 for(const name of filenames)for(const [flag,suffix] of [['--cpu','js'],['--wgsl','wgsl']]) {
  const run=spawnSync(process.execPath,[path.join(ROOT,'quines/'+name+'.js'),flag],{encoding:'utf8',timeout:10000,maxBuffer:2000000});
  assert.equal(run.status,0,run.stderr);assert.equal(run.stdout,read('quines/'+name+'.'+suffix));
 }
});
const result={schema:'five-quines-verification/v1',environment:{node:process.version,platform:process.platform,arch:process.arch},
 scope:'JavaScript execution, independent emitter specification, independent Sprout interpreter, CPU image reference. No WGSL compilation or GPU dispatch.',
 passed:true,checks,programs:filenames.map(name=>({name,bytes:Buffer.byteLength(sources[name]),sha256:sha(sources[name])}))};
fs.writeFileSync(path.join(ROOT,'verification/cpu-results.json'),JSON.stringify(result,null,2)+'\n');
console.log(`\n${checks.length} groups passed. GPU execution is a separate test.`);
