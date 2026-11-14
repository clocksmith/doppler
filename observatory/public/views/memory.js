import { $,h } from '../viz/dom.js';
import { gap } from '../viz/gap.js';
import { provenanceMark } from '../viz/provenance.js';
import { memoryLandscape } from './witness-data.js';
import { withActivity } from './witness-ui.js';
export function renderMemory({run,selection}) {
  const root=$('memory');
  withActivity(root,run,selection,(activity,position,layer)=>{
    const rows=memoryLandscape(activity,position,layer);
    if(!rows.length){root.replaceChildren(gap({what:'No KV writes, recurrent state or convolution history were captured for this layer. No memory values are inferred from token probabilities.'}));return;}
    const max=Math.max(0,...rows.map(r=>r.value??0));
    const groups=new Map();
    for(const r of rows) { const label=`${r.role} · layer ${r.layer+1}`; if(!groups.has(label))groups.set(label,[]);groups.get(label).push(r); }
    root.replaceChildren(h('p',{},provenanceMark('derived'),' RMS magnitude of recorded memory values. This is a landscape of the latest saved writes and state snapshots in the recording, not a time-exact cache, allocation size, residency, meaning or recall strength.'),
      h('p',{className:'demo-note'},`KV writes show the most recent 64 positions through ${position}; older writes are outside this page. Raw state prefixes may be partial. Prefill state is captured at the chunk boundary, not separately at each token. Scale 0–${max.toPrecision(4)} RMS.`),
      ...[...groups].map(([label,items])=>h('div',{},h('h4',{className:'type-label'},label),h('div',{style:'display:flex;flex-wrap:wrap;gap:3px'},...items.map(r=>h('button',{type:'button',className:'btn btn-small',ariaLabel:`${label}, position ${r.position}: ${r.value===null?'not recorded':r.value.toPrecision(5)+' RMS'}`,title:`${r.count} saved values${r.boundary?' after '+r.boundary:''}`,style:`min-width:28px;min-height:50px;padding-bottom:22px;border-style:${r.value===null?'dashed':'solid'};box-shadow:inset 0 -${r.value===null||!max?0:Math.round(20*r.value/max)}px currentColor`,on:{click:()=>selection.select({position:r.position,layer:r.layer})}},String(r.position)))),...items.filter(r=>r.boundary).map(r=>h('p',{className:'demo-note'},`${r.boundary}; ${r.count} saved raw values; RMS ${r.value===null?'not recorded':r.value.toPrecision(5)}.`)))));
  });
}
