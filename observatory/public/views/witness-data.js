// Read-only adapters for bounded numerical activity recordings.
const MAX_VALUES = 2 ** 22;
const cache = new WeakMap();
export function numbers(p) {
  if (Array.isArray(p) || ArrayBuffer.isView(p)) { if (p.length > MAX_VALUES) throw new Error('Capture exceeds view budget.'); return p; }
  if (!p) return [];
  if (cache.has(p)) return cache.get(p);
  const stride = p.dtype === 'f64' ? 8 : 4;
  if (p.encoding !== 'base64-le' || !['f32', 'f64'].includes(p.dtype) || !Number.isSafeInteger(p.length) || p.length < 0 || p.length > MAX_VALUES || typeof p.data !== 'string' || p.data.length !== 4 * Math.ceil(p.length * stride / 3)) throw new Error('Malformed or oversized numerical capture.');
  const binary = atob(p.data);
  if (binary.length !== p.length * stride) throw new Error('Truncated numerical capture.');
  const view = new DataView(Uint8Array.from(binary, c => c.charCodeAt(0)).buffer);
  const result = Array.from({ length:p.length }, (_,i) => stride === 8 ? view.getFloat64(i*stride,true) : view.getFloat32(i*stride,true));
  cache.set(p,result); return result;
}
export function activityOf(run) {
  const trace = run.activityRecording?.trace ?? run.activityTrace;
  if (!trace?.activity) return null;
  // Runs write the hash as "sha256:<hex>", activity traces as bare hex; compare the digests.
  const digestOf = (hash) => String(hash).replace(/^sha256:/, '');
  if (trace.identity?.manifestHash && run.model?.manifestHash && digestOf(trace.identity.manifestHash) !== digestOf(run.model.manifestHash)) throw new Error('Activity belongs to different model files.');
  const a = trace.activity;
  if (!Array.isArray(a.frames) || a.frames.length > 65536 || !Array.isArray(a.weights) || a.weights.length > 65536) throw new Error('Malformed or oversized activity inventory.');
  return { ...a, identity:trace.identity };
}
export function tokenRow(f,p) {
  if (f?.status !== 'captured' || f.representation === 'aggregated') return [];
  const width=f.fullShape?.[1], row=p-f.tokenStart;
  if (!Number.isSafeInteger(width) || width<1 || row<0 || row>=f.fullShape[0]) return [];
  const values=numbers(f.values);
  return (row+1)*width>values.length ? [] : Array.from(values.slice(row*width,(row+1)*width));
}
export function latestFrame(frames,role,layer,p,time=Infinity) {
  let found;
  for (const f of frames) if (f.role===role && f.layer===layer && f.atMs<=time && p>=f.tokenStart && p<f.tokenStart+(f.tokenIds?.length??0) && (!found || f.atMs>=found.atMs)) found=f;
  return found;
}
export function rms(values) { return !values.length || values.some(v=>typeof v!=='number'||!Number.isFinite(v)) ? null : Math.sqrt(values.reduce((s,v)=>s+v*v,0)/values.length); }
export function attentionWitness(a,p,layer,head=0) {
  const query=latestFrame(a.frames,'attention-query',layer,p);
  if (query?.status!=='captured') return null;
  const {heads,kvHeads,headDim:dim,kvLength:length,scale,softcap,slidingWindow,causal}=query.metadata??{};
  if (![heads,kvHeads,dim,length].every(n=>Number.isSafeInteger(n)&&n>0&&n<=65536) || heads%kvHeads || !Number.isInteger(head) || head<0 || head>=heads || !Number.isFinite(scale) || !Number.isFinite(softcap) || !(slidingWindow===null||Number.isFinite(slidingWindow)) || typeof causal!=='boolean') return null;
  if (heads > 256 || dim > 4096 || length > 4096 || length * dim > 2000000 || length * a.frames.length > 2000000) throw new Error('Attention reconstruction exceeds the view budget; capture fewer keys or layers.');
  const q=tokenRow(query,p).slice(head*dim,(head+1)*dim);
  if (q.length!==dim || q.some(v=>!Number.isFinite(v))) return null;
  const kh=Math.floor(head/(heads/kvHeads)), scores=[];
  for(let k=0;k<length;k++) {
    if((causal&&k>p)||(slidingWindow>0&&k<p-slidingWindow+1)){scores.push(-Infinity);continue;}
    const key=tokenRow(latestFrame(a.frames,'kv-key',layer,k,query.atMs),k).slice(kh*dim,(kh+1)*dim);
    if(key.length!==dim || key.some(v=>!Number.isFinite(v))) return null;
    let score=q.reduce((s,v,i)=>s+v*key[i],0)*scale;
    if(softcap>0) score=softcap*Math.tanh(score/softcap);
    if(!Number.isFinite(score)) return null;
    scores.push(score);
  }
  const max=Math.max(...scores); if(!Number.isFinite(max))return null;
  const exp=scores.map(s=>Math.exp(s-max)), sum=exp.reduce((a,b)=>a+b,0);
  return {probabilities:exp.map(v=>v/sum),heads,query,kvHead:kh};
}
export function gearboxProducts(a,w,p) {
  if(a.identity?.adapters?.length || !['down','ffnDown'].includes(w.role) || w.layout !== 'row' || w.status !== 'captured' || !numbers(w.columnEnergy).length || w.shape?.length!==2 || !Number.isInteger(w.layer))return null;
  const input=tokenRow(latestFrame(a.frames,'ffn-down-input',w.layer,p),p), [rows,cols]=w.shape;
  if(input.length!==cols || input.some(v=>!Number.isFinite(v)))return null;
  if (![rows,cols].every(n=>Number.isSafeInteger(n)&&n>0) || rows*cols>1e12) return null;
  const raw=numbers(w.raw), offset=w.rawOffset;
  if(!Number.isSafeInteger(offset)||offset<0||offset+raw.length>rows*cols)return null;
  const products=Array.from(raw.slice(0,256),(weight,i)=>({index:offset+i,row:Math.floor((offset+i)/cols),column:(offset+i)%cols,weight,input:input[(offset+i)%cols],product:weight*input[(offset+i)%cols]}));
  if(products.some(p=>!Number.isFinite(p.product)))return null;
  return {products,total:rows*cols,saved:raw.length,columns:cols};
}
export function memoryLandscape(a,p,layer=null) {
  const rows=[];
  if (a.frames.length > 8192) throw new Error('Memory inventory exceeds the view budget; capture fewer layers.');
  for(const role of ['kv-key','kv-value','recurrent-memory','convolution-history']) {
    const layers=[...new Set(a.frames.filter(f=>f.role===role&&(layer===null||f.layer===layer)).map(f=>f.layer))].sort((a,b)=>a-b);
    if (layers.length > 64 || layers.length * a.frames.length * Math.min(64,p+1) > 2000000) throw new Error('Memory landscape exceeds the view budget; select a layer.');
    for(const l of layers) if(role.startsWith('kv-')) {
      for(let pos=Math.max(0,p-63);pos<=p;pos++) {const values=tokenRow(latestFrame(a.frames,role,l,pos),pos);rows.push({role,layer:l,position:pos,value:rms(values),count:values.length,boundary:null});}
    } else {
      const f=latestFrame(a.frames,role,l,p), values=f?.status==='captured'?Array.from(numbers(f.values)):[];
      rows.push({role,layer:l,position:p,value:rms(values),count:values.length,boundary:f?`${f.phase} positions ${f.tokenStart}–${f.tokenStart+f.tokenIds.length-1}`:null});
    }
  }
  return rows;
}

// The nearest sequence position, before or after `position`, for which `recorded`
// holds; ties go to the earlier one. Null when none in [0, length) does.
export function nearestRecorded(position, length, recorded) {
  for (let distance = 1; distance < length; distance++) {
    for (const candidate of [position - distance, position + distance]) {
      if (candidate >= 0 && candidate < length && recorded(candidate)) return candidate;
    }
  }
  return null;
}
