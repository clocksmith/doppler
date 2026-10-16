// Projection-shaped graph probe; requested devices are not backend identity evidence.
import fs from 'node:fs/promises';
import http from 'node:http';
import { chromium } from 'playwright';
const server=http.createServer((_request,response)=>response.end('<!doctype html><title>Projection probe</title>'));
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const flags=['--enable-features=WebMachineLearningNeuralNetwork'];
const browser=await chromium.launch({channel:'chrome',headless:true,args:flags});
let result;
try {
  const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}`);
  result=await page.evaluate(async()=>{
    const rows=[];
    for(const requestedDeviceType of ['cpu','gpu','npu']) {
      let context;
      try {
        context=await navigator.ml.createContext({deviceType:requestedDeviceType});
        const M=128,K=1024,N=3584;
        const a=Float32Array.from({length:M*K},(_,i)=>((Math.floor(i/K)%7)-3)/8);
        const b=Float32Array.from({length:K*N},(_,i)=>((i%N)%11-5)/16*(Math.floor(i/N)%4-1));
        const start=performance.now();const builder=new MLGraphBuilder(context);
        const x=builder.input('x',{dataType:'float32',shape:[M,K]});
        const w=builder.constant({dataType:'float32',shape:[K,N]},b);
        const graph=await builder.build({y:builder.matmul(x,w)});
        const preparationMs=performance.now()-start;
        const input=await context.createTensor({dataType:'float32',shape:[M,K],writable:true});
        const output=await context.createTensor({dataType:'float32',shape:[M,N],readable:true});
        const times=[];
        for(let run=0;run<6;run++){
          const began=performance.now();context.writeTensor(input,a);context.dispatch(graph,{x:input},{y:output});
          const values=new Float32Array(await context.readTensor(output));const elapsed=performance.now()-began;
          for(let i=0;i<values.length;i++){
            const expected=((Math.floor(i/N)%7)-3)/8*((i%N)%11-5)/16*(K/2);
            if(values[i]!==expected)throw new Error(`Numerical mismatch at ${i}: ${values[i]} != ${expected}`);
          }
          if(run>1)times.push(elapsed);
        }
        rows.push({requestedDeviceType,M,N,K,weightDtype:'float32',preparationMs,times,
          checkedOutputs:M*N,oracle:'Exact separable dyadic matrix, synthetic values; not model weights',
          timing:'write input + dispatch + complete output readback; verification excluded',
          backendIdentity:'Unverified; browser may use a different underlying accelerator'});
        input.destroy();output.destroy();graph.destroy();
      }catch(error){rows.push({requestedDeviceType,error:String(error)});}
      finally{context?.destroy();}
    }
    return {userAgent:navigator.userAgent,rows};
  });
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
result={flags,browserVersion:browser.version(),scope:'Graph capability and diagnostic latency only; no Qwen WebNN backend',...result};
await fs.writeFile(new URL('./webnn-projection.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
