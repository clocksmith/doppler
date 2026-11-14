import { h } from '../viz/dom.js';
import { gap } from '../viz/gap.js';
import { activityOf } from './witness-data.js';
export function svg(tag,attrs={},...children) {
  const n=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));
  n.append(...children);return n;
}
export function withActivity(root,run,selection,render) {
  if(!root)return;
  const {position,layer}=selection.get();
  if(position===null){root.replaceChildren(h('p',{className:'demo-note'},'Select a token to inspect its recorded activity.'));return;}
  try {
    const activity=activityOf(run);
    if(!activity){root.replaceChildren(gap({what:'This run did not retain detailed activity. Open a run saved with grids, or record detailed activity.'}));return;}
    render(activity,position,layer);
  } catch(error){root.replaceChildren(gap({what:`Cannot display this capture: ${error.message}`}));}
}
export function layerPicker(activity,role,selected,selection) {
  const layers=[...new Set(activity.frames.filter(f=>f.role===role).map(f=>f.layer))].filter(Number.isInteger).sort((a,b)=>a-b);
  return h('label',{},'Layer ',h('select',{ariaLabel:'Activity layer',value:String(selected),on:{change:e=>selection.select({layer:Number(e.target.value)})}},...layers.map(l=>h('option',{value:String(l),selected:l===selected},String(l+1)))));
}

// Offered beside a gap: the closest token (and, if needed, layer) that was recorded.
// `target` is { position, layer } or null; positions are sequence positions.
export function nearestOffer(run,selection,target,what) {
  if(!target) return h('p',{className:'demo-note'},`No token in this run recorded ${what}.`);
  const prompt=run.inputTokenIds?.length??0, index=target.position-prompt, token=index>=0?run.tokens?.[index]??null:null;
  const label=token?`word ${index+1}, "${token.text??''}"`:`prompt position ${target.position}`;
  return h('p',{},h('button',{type:'button',className:'btn btn-small',on:{click:()=>selection.select(target)}},`Go to the nearest recorded token: ${label}${target.layer!==undefined?`, layer ${target.layer+1}`:''}`));
}
export const sequenceLength=(run)=>(run.inputTokenIds?.length??0)+(run.tokens?.length??0);
