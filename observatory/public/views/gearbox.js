import { $,h } from '../viz/dom.js';
import { gap } from '../viz/gap.js';
import { provenanceMark } from '../viz/provenance.js';
import { gearboxProducts,nearestRecorded } from './witness-data.js';
import { withActivity,svg,nearestOffer,sequenceLength } from './witness-ui.js';
const weights=new Map();
export function renderGearbox(context) {
  const {run,selection}=context,root=$('gearbox');
  withActivity(root,run,selection,(activity,position,layer)=>{
    const candidates=activity.weights.filter(w=>['down','ffnDown'].includes(w.role)&&w.layout==='row'&&w.shape?.length===2&&(layer===null||w.layer===layer));
    const weight=candidates.find(w=>w.id===weights.get(run.id))??candidates.find(w=>gearboxProducts(activity,w,position)?.products.length);
    const result=weight?gearboxProducts(activity,weight,position):null;
    const picker=h('select',{ariaLabel:'Weight tensor',on:{change:e=>{weights.set(run.id,e.target.value);renderGearbox(context);}}},...candidates.map(w=>h('option',{value:w.id,selected:w.id===weight?.id},w.id)));
    if(!result?.products.length){
      // The closest token with a saved weight window and its FFN-down input, at the chosen layer if one is chosen.
      const pool=activity.weights.filter(w=>['down','ffnDown'].includes(w.role)&&w.layout==='row'&&w.shape?.length===2&&(layer===null||w.layer===layer));
      const recorded=(p)=>pool.some(w=>{try{return Boolean(gearboxProducts(activity,w,p)?.products.length);}catch{return false;}});
      const near=activity.identity?.adapters?.length?null:nearestRecorded(position,sequenceLength(run),recorded);
      root.replaceChildren(...(candidates.length ? [h('label',{},'Tensor ',picker)] : []),gap({what:activity.identity?.adapters?.length?'Effective adapter weights were not captured; base-only products would be incomplete.':'No matched saved weight window and FFN-down projection input at this token and layer. A weight inventory alone cannot establish connection products.'}),
        activity.identity?.adapters?.length?null:nearestOffer(run,selection,near===null?null:{position:near},`connection products${layer===null?'':` at layer ${layer+1}`}`));return;}

    const largest=Math.max(0,...result.products.map(p=>Math.abs(p.product))),chart=svg('svg',{viewBox:'0 0 800 180',role:'img','aria-label':'Signed weight times input products in the saved tensor window',style:'width:100%;max-height:230px'});
    chart.append(svg('line',{x1:0,y1:90,x2:800,y2:90,stroke:'currentColor'}));
    for(const [i,p] of result.products.entries()) {const height=largest?Math.abs(p.product)/largest*80:0,width=800/result.products.length;const rect=svg('rect',{x:i*width,y:p.product>=0?90-height:90,width:Math.max(.5,width-.5),height,fill:'currentColor','fill-opacity':p.product>=0?.8:.35});rect.append(svg('title',{},`row ${p.row}, column ${p.column}: ${p.weight} × ${p.input} = ${p.product}`));chart.append(rect);}
    root.replaceChildren(h('p',{},provenanceMark('reconstructed'),' Stored weight × captured FFN-down input. These mathematical products are not kernel-rounded products or proof of causation.'),h('label',{},'Tensor ',picker),chart,h('p',{className:'demo-note'},`Layer ${weight.layer+1}, token position ${position}. Showing ${result.products.length} connections from ${result.saved} saved of ${result.total} total. Above zero is positive; below is negative. Vertical scale ±${largest.toPrecision(4)}.`),h('details',{},h('summary',{},'Inspect the first 32 connection products'),h('div',{className:'site-scroll'},h('table',{},h('thead',{},h('tr',{},...['Row','Column','Weight','Input','Product'].map(t=>h('th',{},t)))),h('tbody',{},...result.products.slice(0,32).map(p=>h('tr',{},...['row','column','weight','input','product'].map(k=>h('td',{},String(p[k]))))))))));
  });
}
