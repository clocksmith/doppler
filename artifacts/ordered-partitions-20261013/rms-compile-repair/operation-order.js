import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
const trace=fn=>{let id=0;const events=[];const op=(name,...args)=>{const value='v'+id++;events.push({name,args,value});return value;};return{result:fn(op),events};};
const before=trace(op=>{const mean=op('div','squares','size');const denominator=op('sqrt',op('add',mean,'eps'));return op('div','one',denominator);});
const after=trace(op=>{let numerator='squares',divisor='size',inverse;for(let stage=0;stage<2;stage++){const quotient=op('div',numerator,divisor);if(stage===0){divisor=op('sqrt',op('add',quotient,'eps'));numerator='one';}else inverse=quotient;}return inverse;});
assert.deepEqual(after,before);writeFileSync(new URL('./operation-order.json',import.meta.url),JSON.stringify({scope:'Symbolic scalar operands and evaluation order, not GPU numerical proof',before,after,passed:true},null,2)+'\n');
console.log('RMS scalar operands and evaluation order preserved');
