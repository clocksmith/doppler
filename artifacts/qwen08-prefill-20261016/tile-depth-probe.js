// Fixed divisible-by-eight matrix probe, not a selectable Doppler runtime kernel.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
console.log("provider imported");
const gpu = create([]);
// Dawn's Node async event pump borrows its instance; retain it through process exit.
globalThis.projectionProbeGpu = gpu;
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice({requiredFeatures:['shader-f16'],requiredLimits:{maxComputeWorkgroupStorageSize:adapter.limits.maxComputeWorkgroupStorageSize}});
console.log('device ready');
const result={scope:'Standalone F32 accumulation, F16 weight projection; not model qualification',
  adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture},
  scopeDetail:'Registered tiled matmul load mapping; unchanged arithmetic',
  tolerance:{absolute:1e-3,relative:1e-3},rows:[]};
try {
  const variants=[['before','./matmul-tiled-before.wgsl',64],['k16','./matmul-depth-rejected.wgsl',64,16],['k32','./matmul-depth-rejected.wgsl',64,32],['k64','./matmul-depth-rejected.wgsl',64,64]];
  const pipelines=[];
  for(const [name,file,tile,tileK] of variants){
    console.log(`compile ${name}`);
    const code=await fs.readFile(new URL(file,import.meta.url),'utf8');
    device.pushErrorScope('validation');
    const start=performance.now();const module=device.createShaderModule({code});
    const info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(x=>x.type==='error'),[]);
    const pipeline=device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'main',constants:tileK ? {TILE_K:tileK} : {}}});
    assert.equal(await device.popErrorScope(),null,'Pipeline validation');
    pipelines.push({name,tile,pipeline,compileMs:performance.now()-start,sha256:createHash('sha256').update(code).digest('hex')});
  }
  for(const [M,N,K] of [[65,67,19],[128,3584,1024],[1024,3584,1024]]){
    const buffers=[];
    const make=(size,usage)=>{const b=device.createBuffer({size,usage});buffers.push(b);return b;};
    try {
      const a=Float32Array.from({length:M*K},(_,i)=>Math.sin(i*.013)*.125);
      const b=new Uint16Array(Math.ceil(N*K/2)*2), bValues=new Float32Array(N*K);
      for(let i=0;i<b.length;i++){const magnitude=1+(i*17%31); b[i]=((i%3===0?1:0)<<15)|(11<<10)|(magnitude<<5); bValues[i]=(i%3===0?-1:1)*(1+magnitude/32)/16;}
      const uniform=make(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
      const A=make(a.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
      const B=make(b.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
      const C=make(M*N*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
      const readback=make(M*N*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(uniform,0,new Uint32Array([M,N,K,0x3f800000,1,0,0,0]));
      device.queue.writeBuffer(A,0,a);device.queue.writeBuffer(B,0,b);await device.queue.onSubmittedWorkDone();
      let baseline;
      for(const {name,tile,pipeline,compileMs,sha256} of pipelines){
        const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[uniform,A,B,C].map((buffer,binding)=>({binding,resource:{buffer}}))});
        device.pushErrorScope('validation');
        device.queue.writeBuffer(C,0,new Float32Array(M*N).fill(NaN));
        assert.equal(await device.popErrorScope(),null,'Output poisoning validation');
        const times=[];let values;
        for(let run=0;run<4;run++){
          device.pushErrorScope('validation');
          const start=performance.now();const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();
          pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(M/tile),Math.ceil(N/tile));pass.end();
          encoder.copyBufferToBuffer(C,0,readback,0,M*N*4);device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();await readback.mapAsync(GPUMapMode.READ);values=new Float32Array(readback.getMappedRange()).slice();readback.unmap();
          if(run)times.push(performance.now()-start);
          assert.equal(await device.popErrorScope(),null,'Dispatch validation');
        }
        let maxAbs=0,maxRefAbs=0;
        if(!baseline)baseline=values;
        for(let i=0;i<values.length;i++){assert.ok(Number.isFinite(values[i]));const delta=Math.abs(values[i]-baseline[i]);maxAbs=Math.max(maxAbs,delta);assert.ok(delta<=.001+.001*Math.abs(baseline[i]),`${name} index ${i}`);}
        for(let t=0;t<64;t++){const row=t*37%M,col=t*197%N;let reference=0;for(let k=0;k<K;k++)reference+=a[row*K+k]*bValues[col*K+k];const delta=Math.abs(values[row*N+col]-reference);maxRefAbs=Math.max(maxRefAbs,delta);assert.ok(delta<=.001+.001*Math.abs(reference));}
        result.rows.push({M,N,K,name,compileMs,sha256,times,maxAbsVsScalar:maxAbs,maxAbsVsReference:maxRefAbs,checkedOutputs:values.length,referenceSamples:64});
        await fs.writeFile(new URL('./tile-depth-probe.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
        console.log(JSON.stringify(result.rows.at(-1)));
      }
    }finally{buffers.forEach(x=>x.destroy());}
  }
}catch(error){result.error={message:error.message,stack:error.stack};process.exitCode=1;}
finally{device.destroy();await fs.writeFile(new URL('./tile-depth-probe.json',import.meta.url),JSON.stringify(result,null,2)+'\n');}
