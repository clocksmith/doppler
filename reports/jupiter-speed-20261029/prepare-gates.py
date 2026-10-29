"""Build isolated, untimed full-logit and dispatch-audit harness."""
from pathlib import Path
import json,shutil,hashlib
root=Path(__file__).resolve().parent
receipt=json.loads((root/'runs/receipt.json').read_text());assert receipt.get('complete')
contract=json.loads((root/'contract.json').read_text())
teachers=[]
for prompt in contract['prompts']:
 row=next(s for s in receipt['sessions'] if s['prompt']==prompt['id'] and s['lane']=='baseline')
 teachers.append({'id':prompt['id'],'promptIds':row['promptIds'],'outputIds':row['outputIds']})
(root/'teachers.json').write_text(json.dumps(teachers,indent=2)+'\n')
source=root/'runtime';dest=root/'gate-runtime';shutil.copytree(source,dest,symlinks=True)
p=dest/'src/inference/browser-harness/benchmark-execution.js';s=p.read_text()
audit='''
  // Untimed diagnostic clone only: wrappers preserve commands and pass topology.
  const auditPipelines = new WeakMap(), auditSeen = new Map();
  const auditDescriptor = (descriptor) => ({label: descriptor.label ?? null, entryPoint: descriptor.compute.entryPoint ?? 'main', constants: descriptor.compute.constants ?? {}, moduleLabel: descriptor.compute.module.label ?? null, dispatches: 0});
  const auditSync = GPUDevice.prototype.createComputePipeline;
  GPUDevice.prototype.createComputePipeline = function(descriptor) { const pipeline = auditSync.call(this, descriptor); auditPipelines.set(pipeline, auditDescriptor(descriptor)); return pipeline; };
  const auditAsync = GPUDevice.prototype.createComputePipelineAsync;
  GPUDevice.prototype.createComputePipelineAsync = async function(descriptor) { const pipeline = await auditAsync.call(this, descriptor); auditPipelines.set(pipeline, auditDescriptor(descriptor)); return pipeline; };
  const auditPass = new WeakMap(), auditSet = GPUComputePassEncoder.prototype.setPipeline;
  GPUComputePassEncoder.prototype.setPipeline = function(pipeline) { auditPass.set(this, pipeline); return auditSet.call(this,pipeline); };
  for (const method of ['dispatchWorkgroups','dispatchWorkgroupsIndirect']) {
    const original = GPUComputePassEncoder.prototype[method];
    GPUComputePassEncoder.prototype[method] = function(...args) { const pipeline=auditPass.get(this), row=auditPipelines.get(pipeline); if(row) {row.dispatches++;auditSeen.set(pipeline,row);} return original.apply(this,args); };
  }
'''
anchor='  const harness = await withHarnessPhase(';assert s.count(anchor)==1;s=s.replace(anchor,audit+'\n'+anchor)
hook='''
  // Separate correctness execution: never include these readbacks in speed claims.
  const teachers = TEACHERS;
  const teacher = teachers.find(t => JSON.stringify(t.promptIds) === JSON.stringify(metrics.referenceTranscript.prompt.ids));
  if (!teacher || teacher.outputIds.length !== 64) throw Error('Expected frozen 64-token teacher');
  const gatePipeline = harness.pipeline.generator;
  harness.pipeline.reset();
  const gateOptions = gatePipeline._resolveStepOptions({temperature:0,topK:1,topP:1,repetitionPenalty:1,useChatTemplate:false,benchmark:false});
  const rows = [];
  const saveLogits = (position,v) => {if(!(v instanceof Float32Array))throw Error('Expected full F32 vocabulary');const bytes=new Uint8Array(v.buffer,v.byteOffset,v.byteLength);let str='';for(let i=0;i<bytes.length;i+=16384)str+=String.fromCharCode(...bytes.subarray(i,i+16384));rows.push({position,elements:v.length,base64:btoa(str)});};
  saveLogits(0, await gatePipeline._prefillInputIdsToLogits(teacher.promptIds,gateOptions));
  const ids=teacher.promptIds.slice(), positions=new Set([1,2,4,8,16,32,63]);
  for(let k=1;k<=63;k++) {ids.push(teacher.outputIds[k-1]);const value=await gatePipeline.decodeStepLogits(ids,{temperature:0,topK:1,topP:1,repetitionPenalty:1,useChatTemplate:false});if(positions.has(k))saveLogits(k,value.logits);}
  metrics.fullLogitGate={teacher,rows,scope:'Untimed fixed-teacher materialized-head logits; not fused argmax intermediate values'};
  metrics.dispatchAudit={scope:'Untimed diagnostic including warmup/scored-route execution and teacher-logit replay; no timestamp probes or pass splits',pipelines:[...auditSeen.values()]};
'''.replace('TEACHERS',json.dumps(teachers))
anchor='  const loadTimings = resolvePipelineLoadTimings(harness.pipeline);';assert s.count(anchor)==1;s=s.replace(anchor,hook+'\n'+anchor);p.write_text(s)
(root/'gate-source.json').write_text(json.dumps({'modifiedPath':str(p.relative_to(dest)),'baselineSha256':hashlib.sha256((source/p.relative_to(dest)).read_bytes()).hexdigest(),'gateSha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Diagnostic-only clone; no timing or production-source claim'},indent=2)+'\n')
