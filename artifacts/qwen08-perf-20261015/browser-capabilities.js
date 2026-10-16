import fs from 'node:fs/promises';
import http from 'node:http';
import { chromium } from 'playwright';
const server = http.createServer((_req,res)=>res.end('<!doctype html><title>Local compute capability probe</title>'));
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const results=[];
try {
  for(const flags of [[],['--enable-features=WebMachineLearningNeuralNetwork']]) {
    const browser=await chromium.launch({channel:'chrome',headless:true,args:flags});
    try {
      const page=await browser.newPage();
      await page.goto(`http://127.0.0.1:${server.address().port}`);
      const result=await page.evaluate(async()=>{
        const record={userAgent:navigator.userAgent,secureContext:isSecureContext,webgpu:!!navigator.gpu,webnn:!!navigator.ml,builder:typeof MLGraphBuilder,contexts:[]};
        if(!navigator.ml||typeof MLGraphBuilder!=='function')return record;
        for(const deviceType of ['cpu','gpu','npu']) {
          let context;
          try {
            context=await Promise.race([navigator.ml.createContext({deviceType}),new Promise((_,reject)=>setTimeout(()=>reject(new Error('Context timeout')),10000))]);
            const builder=new MLGraphBuilder(context);
            const descriptor={dataType:'float32',shape:[2,2]};
            const x=builder.input('x',descriptor);
            const w=builder.constant(descriptor,new Float32Array([5,6,7,8]));
            const graph=await builder.build({y:builder.matmul(x,w)});
            const input=await context.createTensor({...descriptor,writable:true});
            const output=await context.createTensor({...descriptor,readable:true});
            context.writeTensor(input,new Float32Array([1,2,3,4]));
            context.dispatch(graph,{x:input},{y:output});
            const values=Array.from(new Float32Array(await context.readTensor(output)));
            record.contexts.push({requestedDeviceType:deviceType,values,correct:JSON.stringify(values)==='[19,22,43,50]',backendIdentity:'Not exposed; requested device type is not proof of actual accelerator'});
            input.destroy();output.destroy();graph.destroy();
          }catch(error){record.contexts.push({requestedDeviceType:deviceType,error:String(error)});}
          finally{context?.destroy();}
        }
        return record;
      });
      results.push({flags,browserVersion:browser.version(),...result});
    }finally{await browser.close();}
  }
}finally{await new Promise(resolve=>server.close(resolve));await fs.writeFile(new URL('./browser-capabilities.json',import.meta.url),JSON.stringify(results,null,2)+'\n');}
console.log(JSON.stringify(results,null,2));
