import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
// Compare exact scalar operation traces for every four-key visibility mask.
const trace=fn=>{const events=[];let id=0;const op=(name,...args)=>{const value='v'+id++;events.push({name,args,value});return value;};const result=fn(op);return{events,result};};
let checked=0;
for(const previous of [false,true])for(let mask=0;mask<16;mask++){
 const before=trace(op=>{let alpha=0;if(previous)alpha=op('exp',op('add','maximum','-next_max'));let sum=op('mul','denominator',alpha);const weights=[];for(let k=0;k<4;k++){weights[k]=0;if(mask&(1<<k)){weights[k]=op('exp',op('add','dot'+k,'-next_max'));sum=op('add',sum,weights[k]);}}return{alpha,sum,weights};});
 const after=trace(op=>{let alpha,sum=0;const weights=[];for(let slot=0;slot<5;slot++){let enabled=previous,score='maximum';if(slot>0){enabled=Boolean(mask&(1<<(slot-1)));if(enabled)score='dot'+(slot-1);}let exponential=0;if(enabled)exponential=op('exp',op('add',score,'-next_max'));if(slot===0){alpha=exponential;sum=op('mul','denominator',alpha);}else{weights[slot-1]=exponential;if(enabled)sum=op('add',sum,exponential);}}return{alpha,sum,weights};});
 assert.deepEqual(after,before);checked++;
}
writeFileSync(new URL('./operation-order.json',import.meta.url),JSON.stringify({scope:'Scalar call operands, ordering and zero masking; no GPU numerical or compiler proof',checked,passed:true},null,2)+'\n');
console.log('All 32 alpha/key visibility traces preserve scalar operation order');
