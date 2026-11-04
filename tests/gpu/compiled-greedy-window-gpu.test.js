import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createGreedySpeculativeSession } from '../../src/gpu/greedy-speculative-frame.js';
import { runLinearAttentionCoreGPU } from '../../src/gpu/kernels/linear-attention-core.js';
import { createLinearAttentionCheckpoint } from '../../src/inference/pipelines/text/linear-attention/state-checkpoint.js';
import { WholeTokenProgram } from '../../src/inference/pipelines/text/generator/whole-token-program.js';
import { createCompiledGreedyWindow } from '../../src/inference/pipelines/text/generator/whole-token-lane.js';

assert.ok((await probeNodeGPU({ installFileFetchShim: true })).ready);
const device=getDevice(), owned=[], errors=[], cases=[];
device.addEventListener('uncapturederror',event=>errors.push(event.error.message));
const H=4,V=32,T=4,K=128,C=384;
const upload=data=>{const b=acquireBuffer(data.byteLength);owned.push(b);uploadData(b,data);return b;};
const values=(n,rate,scale)=>Float32Array.from({length:n},(_,i)=>Math.sin((i+1)*rate)*scale);
const bits=async b=>new Uint32Array(await readBuffer(b,b.size));
const state=()=>({sourceLayout:'gguf-v1',convDim:C,convKernelSize:4,valueDim:K,numVHeads:1,numKHeads:1,
  headKDim:K,headVDim:K,qSize:K,kSize:K,qRep:1,normMode:'per_head',rmsNormEps:1e-6,
  convWeightGPU:upload(values(C*4,.011,.5)),dtBiasGPU:upload(new Float32Array([.2])),aLogGPU:upload(new Float32Array([-.25])),
  normWeightGPU:upload(new Float32Array(K).fill(1)),convStateGPU:upload(values(C*4,.021,.05)),
  recurrentStateGPU:upload(values(K*K,.007,.005)),recurrentStatePrecision:'f32',recurrentStateComponents:1,
  seqLen:0,warnedSeqMismatch:false});
const inputs=[values(T*C,.019,.2),values(T*K,.029,.8),values(T,.11,1.2),values(T,.09,2)]
  .map((a,i)=>createTensor(upload(a),'f32',[T,[C,K,1,1][i]]));
const hidden=upload(values(T*H,.1,.5)), initial=createTensor(upload(new Float32Array([-1,-2,-3,-4])),'f32',[1,H]);
const scores=ids=>{const a=new Float32Array(ids.length*V).fill(-10);ids.forEach((id,i)=>{a[i*V+id]=10;});return upload(a);};
const drafts=[10,11,12].map(id=>scores([id])), target=scores([10,11,12,13]);
const clone=(rec,buffer,rows,width)=>{const b=acquireBuffer(rows*width*4);rec.getEncoder().copyBufferToBuffer(buffer,0,b,0,b.size);return createTensor(b,'f32',[rows,width]);};
const core=register=>({numTokens:T,normGateActivation:'silu',outputDtype:'f32',qkL2NormEps:1e-6,
  runtimeSession:{useLinearAttentionRegisterPrefill:register,useLinearAttentionRegisterDecode:register,useLinearAttentionFusedDecodeCore:false}});
try {
  for(const register of [false,true]){
    const actual=state(), reference=state(), identity=[actual];let pos=0;
    const session=await createGreedySpeculativeSession({owner:actual,hiddenSize:H,vocabSize:V,draftCount:3,draftVocabSize:V,
      refillShapes:[{rows:T,width:H}],initialHidden:initial,initialToken:7,padTokenId:null,
      finitenessAbsThreshold:3.4028234663852886e38,usePreparedDecodeWorkspace:false});
    const cap=upload(new Uint32Array([4]));
    const flat=new WholeTokenProgram(device,{tokenInput:session.carry.token,maxOutputTokens:4});
    const program=()=>({
      async recordDraft(rec,ids,index,previous,j){return{hidden:clone(rec,initial.buffer,1,H),logits:clone(rec,drafts[j],1,V)};},
      async recordTarget(rec){const out=await runLinearAttentionCoreGPU(...inputs,actual,{...core(register),recorder:rec});rec.trackTemporaryBuffer(out.buffer);
        return{hidden:clone(rec,hidden,T,H),logits:clone(rec,target,T,V)};},
      async recordRefill(rec){return[clone(rec,hidden,T,H)];},
      setPrefix(count){actual.seqLen=pos+count;},
    });
    let checkpoint;
    const savedHidden=upload(new Float32Array(H)),savedToken=upload(new Uint32Array(1));
    try {
      for(let i=0;i<4;i++){
        const transaction=await flat.observe(pos,0,attach=>session.begin(program(),4,undefined,{observe:attach,outputLimitBuffer:cap}),()=>identity,
          [{owner:'fixture capacity',min:0,max:128}]);
        for(const token of transaction.tokens){assert.ok(token<V);transaction.publish();}
        await transaction.settle();if(flat.ready)flat.acceptObservedWindow(pos,transaction.tokens.length);pos+=transaction.tokens.length;
        const out=await runLinearAttentionCoreGPU(...inputs,reference,core(register));owned.push(out.buffer);
      }
      assert.equal(flat.ready,true);assert.deepEqual(await bits(actual.recurrentStateGPU),await bits(reference.recurrentStateGPU));
      checkpoint=createLinearAttentionCheckpoint({layers:new Map([[0,actual]])});
      for(const scenario of ['full','partial1','partial2','partial3','none','cap1','cap2','cap3','invalid-cap','callback-failure','map-failure']){
        const beforeConv=await bits(actual.convStateGPU),beforeState=await bits(actual.recurrentStateGPU);
        const beforeHidden=await bits(session.carry.hidden.buffer),beforeToken=await bits(session.carry.token);
        const beginPos=pos;let submitted=false;
        uploadData(cap,new Uint32Array([scenario==='invalid-cap'?0:scenario.startsWith('cap')?Number(scenario.slice(3)):4]));
        const capture=encoder=>{checkpoint.recordCapture(encoder);encoder.copyBufferToBuffer(session.carry.hidden.buffer,0,savedHidden,0,H*4);encoder.copyBufferToBuffer(session.carry.token,0,savedToken,0,4);};
        const restore=async()=>{await device.queue.onSubmittedWorkDone();const encoder=device.createCommandEncoder();checkpoint.recordRestore(encoder);
          encoder.copyBufferToBuffer(savedHidden,0,session.carry.hidden.buffer,0,H*4);encoder.copyBufferToBuffer(savedToken,0,session.carry.token,0,4);
          device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();flat.resumeAfterRestore(beginPos);};
        const ports={identity,recordPrefix:capture,onSubmitted(){submitted=true;if(scenario==='callback-failure')throw new Error('injected callback');}};
        const stage=flat.staging[0],map=stage.mapAsync;
        let delivered=scenario==='full'?4:scenario.startsWith('partial')?Number(scenario.slice(7)):scenario.startsWith('cap')?Number(scenario.slice(3)):0;
        if(scenario==='map-failure')stage.mapAsync=()=>Promise.reject(new Error('injected map failure'));
        try {
          if(['invalid-cap','callback-failure','map-failure'].includes(scenario)){
            await assert.rejects(flat.runWindow(pos,ports),/output count|injected/);assert.equal(submitted,true);
          }else{
            const words=await flat.runWindow(pos,ports);assert.equal(words[2],0);assert.equal(words[words.length-4],0);
            assert.deepEqual(Array.from(words.slice(5,5+words[0])),[10,11,12,13].slice(0,words[0]));
            assert.equal(words[0],scenario.startsWith('cap')?delivered:4);
          }
        } finally {stage.mapAsync=map;}
        if(delivered!==4){
          await restore();
          if(delivered){
            uploadData(cap,new Uint32Array([delivered]));
            const result=await flat.runWindow(pos,{identity,recordPrefix(){},onSubmitted(){}});
            assert.equal(result[0],delivered);assert.equal(result[2],0);
          }
        }
        if(delivered){const out=await runLinearAttentionCoreGPU(...inputs,reference,{...core(register),numTokens:delivered});owned.push(out.buffer);}
        else{
          assert.deepEqual(await bits(actual.convStateGPU),beforeConv);assert.deepEqual(await bits(actual.recurrentStateGPU),beforeState);
          assert.deepEqual(await bits(session.carry.hidden.buffer),beforeHidden);assert.deepEqual(await bits(session.carry.token),beforeToken);
        }
        assert.deepEqual(await bits(actual.convStateGPU),await bits(reference.convStateGPU),scenario);
        assert.deepEqual(await bits(actual.recurrentStateGPU),await bits(reference.recurrentStateGPU),scenario);
        actual.seqLen=beginPos+delivered;pos+=delivered;
        cases.push({register,scenario,delivered,exactState:true});
      }
    } finally {await device.queue.onSubmittedWorkDone();checkpoint?.close();flat.destroy();await session.dispose();}
  }
  for(const register of [false,true]){
    const actual=state(),reference=state(),identity=[actual];let pos=0,activeSignal,failCursor=false;
    const session=await createGreedySpeculativeSession({owner:actual,hiddenSize:H,vocabSize:V,draftCount:3,draftVocabSize:V,
      refillShapes:[{rows:T,width:H}],initialHidden:initial,initialToken:7,padTokenId:null,
      finitenessAbsThreshold:3.4028234663852886e38,usePreparedDecodeWorkspace:false});
    const staging=[],create=device.createBuffer;
    device.createBuffer=function(desc){const buffer=create.call(this,desc);if(desc.label?.startsWith('whole_token_readback_'))staging.push(buffer);return buffer;};
    const controller=createCompiledGreedyWindow({device,session,runtime:{layers:new Map([[0,actual]])},draftCount:3,vocabSize:V,hiddenSize:H,
      identityOf:()=>identity,positionLimits:[{owner:'controller fixture capacity',min:0,max:256}]});
    const program={
      async recordDraft(rec,ids,index,previous,j){return{hidden:clone(rec,initial.buffer,1,H),logits:clone(rec,drafts[j],1,V)};},
      async recordTarget(rec){const out=await runLinearAttentionCoreGPU(...inputs,actual,{...core(register),recorder:rec});rec.trackTemporaryBuffer(out.buffer);
        return{hidden:clone(rec,hidden,T,H),logits:clone(rec,target,T,V)};},
      async recordRefill(rec){return[clone(rec,hidden,T,H)];},
      setPrefix(count){actual.seqLen=pos+count;},
      setReplayedPrefix(count){actual.seqLen=pos+count;activeSignal?.abort(new Error('cancel after GPU submission'));if(failCursor){failCursor=false;throw new Error('injected cursor failure');}},
    };
    try{
      for(let i=0;i<4;i++){
        const transaction=await controller.begin(program,pos,4);
        await assert.rejects(controller.beginGeneration(initial,7),/live idle/);
        for(const token of transaction.tokens){assert.ok(token<V);transaction.publish();}
        await transaction.settle();pos+=transaction.tokens.length;
        const out=await runLinearAttentionCoreGPU(...inputs,reference,core(register));owned.push(out.buffer);
      }
      device.createBuffer=create;assert.equal(controller.ready,true);assert.ok(controller.stats.privateBytes>0);
      for(const scenario of ['full','partial1','partial2','partial3','none','cap1','cap2','cap3','abort-before','abort-after','cursor-failure','map-failure']){
        let delivered=scenario==='full'?4:scenario.startsWith('partial')?Number(scenario.slice(7)):scenario.startsWith('cap')?Number(scenario.slice(3)):0;
        const limit=scenario.startsWith('cap')?delivered:4,signal=new AbortController();
        const beforeHidden=await bits(session.carry.hidden.buffer),beforeToken=await bits(session.carry.token);
        const stage=staging[0],map=stage.mapAsync;
        if(scenario==='abort-before')signal.abort(new Error('cancel before submit'));
        activeSignal=scenario==='abort-after'?signal:null;failCursor=scenario==='cursor-failure';
        if(scenario==='map-failure')stage.mapAsync=()=>Promise.reject(new Error('injected map failure'));
        try{
          if(['abort-before','abort-after','cursor-failure','map-failure'].includes(scenario))await assert.rejects(controller.begin(program,pos,limit,signal.signal),/cancel|injected/);
          else{
            const transaction=await controller.begin(program,pos,limit,signal.signal);
            assert.deepEqual(transaction.tokens,[10,11,12,13].slice(0,limit));
            await assert.rejects(controller.begin(program,pos,4),/live idle/);
            await assert.rejects(controller.dispose(),/Settle/);
            for(let j=0;j<delivered;j++)transaction.publish();
            const settlement=transaction.settle();assert.equal(transaction.settle(),settlement);await settlement;
            assert.throws(()=>transaction.publish(),/publication/);
          }
        }finally{stage.mapAsync=map;activeSignal=null;}
        if(delivered){const out=await runLinearAttentionCoreGPU(...inputs,reference,{...core(register),numTokens:delivered});owned.push(out.buffer);}
        else{assert.deepEqual(await bits(session.carry.hidden.buffer),beforeHidden);assert.deepEqual(await bits(session.carry.token),beforeToken);}
        assert.deepEqual(await bits(actual.convStateGPU),await bits(reference.convStateGPU),scenario);
        assert.deepEqual(await bits(actual.recurrentStateGPU),await bits(reference.recurrentStateGPU),scenario);
        pos+=delivered;assert.equal(actual.seqLen,pos);
        cases.push({controller:true,register,scenario,delivered,exactState:true});
      }
      // A new conversation retains bindings, but must verify a dry-recorded frame
      // before replay. Verification must not submit tentative model updates.
      const before=await bits(actual.recurrentStateGPU);
      const submit=device.queue.submit;let submits=0;
      device.queue.submit=function(commands){submits++;return submit.call(this,commands);};
      try{
        await controller.beginGeneration(initial,7);assert.equal(submits,1,'Only carry initialization submits');
        const transaction=await controller.begin(program,pos,4);
        try{
          assert.equal(controller.stats.verifiedAhead,true);assert.equal(controller.stats.reused,true);
          assert.equal(submits,2,'Dry verification does not submit the layer computation');
        }finally{await transaction.settle();}
        assert.deepEqual(await bits(actual.recurrentStateGPU),before,'Zero publication restores new generation state');
        assert.deepEqual(Array.from(await bits(session.carry.token)).slice(0,1),[7]);
        cases.push({controller:true,register,scenario:'verified-generation-reuse',exactState:true});
      }finally{device.queue.submit=submit;}
      // A changed execution identity invalidates reuse before any cached commands run.
      identity[0]={};
      await controller.beginGeneration(initial,7);
      const changed=await controller.begin(program,pos,4);
      try{assert.equal(controller.ready,false);assert.equal(controller.stats.reused,false);}
      finally{await changed.settle();}
      cases.push({controller:true,register,scenario:'changed-plan-reobserved',exactState:true});
    }finally{device.createBuffer=create;await controller.dispose();await controller.dispose();await session.dispose();}
  }
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,scope:'Physical compiled-window prefix and failure recovery; no model or speed claim',cases}));
} finally {await device.queue.onSubmittedWorkDone();for(const b of owned)releaseBuffer(b);destroyDevice();await releaseNodeWebGPU();}
