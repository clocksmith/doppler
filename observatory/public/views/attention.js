import { $,h,percent } from '../viz/dom.js';
import { gap } from '../viz/gap.js';
import { provenanceMark } from '../viz/provenance.js';
import { attentionWitness,nearestRecorded } from './witness-data.js';
import { svg,withActivity,layerPicker,nearestOffer,sequenceLength } from './witness-ui.js';
const heads=new Map();
export function renderAttention(context) {
  const {run,selection}=context,root=$('attention');
  withActivity(root,run,selection,(activity,position,layer)=>{
    const chosen=layer??activity.frames.find(f=>f.role==='attention-query')?.layer;
    const key=`${run.id}:${chosen}`,head=heads.get(key)??0;
    const witness=attentionWitness(activity,position,chosen,head);
    const controls=layerPicker(activity,'attention-query',chosen,selection);
    if(!witness){
      // The closest recorded query: at this layer first, then at any recorded layer.
      const recordedAt=(l)=>(p)=>{try{return attentionWitness(activity,p,l,0)!==null;}catch{return false;}};
      const length=sequenceLength(run);
      let target=null;
      const here=nearestRecorded(position,length,recordedAt(chosen));
      if(here!==null) target={position:here};
      else for(const l of [...new Set(activity.frames.filter(f=>f.role==='attention-query').map(f=>f.layer))]){
        const p=recordedAt(l)(position)?position:nearestRecorded(position,length,recordedAt(l));
        if(p!==null&&(!target||Math.abs(p-position)<Math.abs(target.position-position))) target={position:p,layer:l};
      }
      root.replaceChildren(controls,gap({what:'A complete query and all unmasked keys were not captured at this position and layer. Partial keys cannot define attention probabilities.'}),nearestOffer(run,selection,target,'a complete attention query'));return;}
    const picker=h('select',{ariaLabel:'Attention head',on:{change:e=>{heads.set(key,Number(e.target.value));renderAttention(context);}}},...Array.from({length:witness.heads},(_,i)=>h('option',{value:String(i),selected:i===head},String(i+1))));
    const ranked=witness.probabilities.map((probability,position)=>({probability,position})).filter(x=>x.probability>0).sort((a,b)=>b.probability-a.probability);
    const displayed=ranked.slice(0,24).sort((a,b)=>a.position-b.position), omitted=ranked.slice(24).reduce((s,x)=>s+x.probability,0);
    const chart=svg('svg',{viewBox:'0 0 800 250',role:'img','aria-label':'Reconstructed attention arcs, strongest 24 keys',style:'width:100%;max-height:300px;overflow:visible'});
    chart.append(svg('line',{x1:20,y1:210,x2:780,y2:210,stroke:'currentColor','stroke-opacity':'.25'}));
    const n=witness.probabilities.length,x=p=>20+760*p/Math.max(1,n-1),qx=x(position);
    for(const item of displayed){const kx=x(item.position),height=40+Math.abs(qx-kx)*.2;
      const path=svg('path',{d:`M ${qx} 210 Q ${(qx+kx)/2} ${210-height} ${kx} 210`,fill:'none',stroke:'currentColor','stroke-width':1+8*item.probability,'stroke-opacity':.25+.75*item.probability});
      path.append(svg('title',{},`Head ${head+1} → position ${item.position}: ${percent(item.probability)}`));chart.append(path);
    }
    chart.append(svg('circle',{cx:qx,cy:210,r:6,fill:'currentColor'}),svg('text',{x:qx,y:235,'text-anchor':'middle',fill:'currentColor'},`query ${position}`));
    root.replaceChildren(h('p',{},provenanceMark('reconstructed'),' CPU softmax of captured Q · K, with recorded scale, grouped-query head mapping, causal/window mask and softcap. Not kernel-bit-exact, and not causal attribution.'),h('div',{className:'site-controls'},controls,h('label',{},' Head ',picker)),chart,
      h('p',{className:'demo-note'},`KV head ${witness.kvHead+1}. Strongest ${displayed.length} keys shown; omitted mass ${percent(omitted)}. Full distribution is not renormalized for display.`),
      h('div',{className:'site-scroll'},h('table',{},h('thead',{},h('tr',{},h('th',{},'Key position'),h('th',{},'Attention'))),h('tbody',{},...ranked.slice(0,24).map(item=>h('tr',{},h('td',{},h('button',{type:'button',className:'btn btn-small',on:{click:()=>selection.select({position:item.position})}},String(item.position))),h('td',{},percent(item.probability))))))));
  });
}
