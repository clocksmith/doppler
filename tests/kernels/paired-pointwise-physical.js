// Actual public wrapper dispatch; no model or runtime policy changes.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';
const destination = process.argv[2];
if (!destination) throw new Error('Provide receipt destination.');
await mkdir(dirname(destination), { recursive: true });
const root = resolve(fileURLToPath(new URL('../../', import.meta.url)));
const receipt = { schema: 'doppler-paired-pointwise-wrapper-physical-v1', sources: {},
  scope: 'Actual immediate/recorded residual, scale, multiply and gated SiLU/sigmoid wrappers against independent F64 equations; no model qualification.' };
for (const path of ['src/gpu/kernels/residual.js', 'src/gpu/kernels/scale.js', 'src/gpu/kernels/silu.js', 'src/rules/generated/rule-bundle.json', 'src/gpu/tensor.js', 'src/gpu/kernels/paired-activation.js', 'src/gpu/kernels/paired_add.wgsl', 'src/gpu/kernels/paired_scale.wgsl', 'src/gpu/kernels/paired_scale_inplace.wgsl', 'src/gpu/kernels/paired_multiply.wgsl', 'src/gpu/kernels/paired_silu_unary.wgsl', 'src/gpu/kernels/paired_silu.wgsl', 'src/gpu/kernels/paired_sigmoid.wgsl', 'src/config/kernels/registry.json']) {
  receipt.sources[path] = createHash('sha256').update(await readFile(resolve(root, path))).digest('hex');
}
const server = createServer(async (req, res) => {
  try {
    const path = resolve(root, '.' + new URL(req.url, 'http://localhost').pathname);
    if (path !== root && !path.startsWith(root + sep)) { res.writeHead(403); res.end(); return; }
    if (path === root) { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html>'); return; }
    res.setHeader('Content-Type', extname(path) === '.js' ? 'text/javascript' : extname(path) === '.json' ? 'application/json' : 'text/plain');
    res.end(await readFile(path));
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu',
    ...(process.platform === 'darwin' ? ['--use-angle=metal'] : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'])] });
  const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}/`);
  receipt.result = await page.evaluate(async () => {
    const { setDevice, destroyDevice } = await import('/src/gpu/device.js');
    const { createTensor } = await import('/src/gpu/tensor.js');
    const { CommandRecorder } = await import('/src/gpu/command-recorder.js');
    const { runResidualAdd, recordResidualAdd } = await import('/src/gpu/kernels/residual.js');
    const { runScale, recordScale } = await import('/src/gpu/kernels/scale.js');
    const { runSiLU, recordSiLU } = await import('/src/gpu/kernels/silu.js');
    const { executePairedMultiply } = await import('/src/gpu/kernels/paired-activation.js');
    const { readBufferSlice, releaseBuffer, destroyBufferPool, getBufferPool } = await import('/src/memory/buffer-pool.js');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Physical adapter required.');
    const device = await adapter.requestDevice(); setDevice(device, { platformConfig: null });
    const errors = [], rows = [], owned = new Set(), pooled = new Set();
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const make = (bytes, upload) => {
      const buffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      owned.add(buffer); if (upload) device.queue.writeBuffer(buffer, 0, upload); return buffer;
    };
    const representation = { precision:'two-f32', components:2, elementStrideBytes:8 };
    const high = [-1000,-104,-40,-16,-1,-.125,0,.125,1,16,40,104,1000];
    const input = new Float32Array(high.flatMap((value,i) => [value, value === 0 ? 0 : (i%2 ? -1 : 1) * 2**-30]));
    const second = new Float32Array(high.flatMap((value,i) => [i%2 ? -1.25 : 1.25, 2**-28]));
    const count = high.length, bytes = count * 8, scale = 1 + 2**-26;
    const pair = data => createTensor(make(data.byteLength,data),'f32',[1,count],'pointwise_fixture',representation);
    const sigmoid = x => x >= 0 ? 1/(1+Math.exp(-x)) : Math.exp(x)/(1+Math.exp(x));
    const tolerance = 6e-12;
    try {
      for (const operation of ['add','scale','multiply','silu','sigmoid','silu_unary']) {
        for (const recorded of [false,true]) for (const borrowed of [false,true]) {
          const a = pair(input), b = pair(second);
          const recorder = recorded ? new CommandRecorder(device,'paired_pointwise_control') : null;
          const options = borrowed ? { outputBuffer:make(bytes) } : {};
          let output;
          try {
            if (operation === 'add') output = recorded ? await recordResidualAdd(recorder,a,b,count,{...options,outputScale:scale}) : await runResidualAdd(a,b,count,{...options,outputScale:scale});
            if (operation === 'scale') output = recorded ? await recordScale(recorder,a,scale,{...options,count}) : await runScale(a,scale,{...options,count});
            if (operation === 'multiply') output = await executePairedMultiply(recorder,a,b,count,options);
            if (operation === 'silu' || operation === 'sigmoid' || operation === 'silu_unary') {
              const unary = operation === 'silu_unary';
              const gateOptions = {...options,size:count,gate:unary?null:a,inputActivation:unary?'silu':'identity',gateActivation:unary?'silu':operation,swigluLimit:null};
              output = recorded ? await recordSiLU(recorder,unary?a:b,gateOptions) : await runSiLU(unary?a:b,gateOptions);
            }
            if (!borrowed) pooled.add(output.buffer);
            if (recorder) await recorder.submitAndWait();
            const actual = Array.from(new Float32Array(await readBufferSlice(output.buffer,0,bytes)));
            let maxError=0;
            for(let i=0;i<count;i++) {
              const x=input[i*2]+input[i*2+1],y=second[i*2]+second[i*2+1];
              const expected=operation==='add'?(x+y)*scale:operation==='scale'?x*scale:operation==='multiply'?x*y:operation==='silu'?x*sigmoid(x)*y:operation==='silu_unary'?x*sigmoid(x):sigmoid(x)*y;
              const error=Math.abs(actual[i*2]+actual[i*2+1]-expected)/Math.max(1,Math.abs(expected));
              if(!Number.isFinite(error)||error>tolerance) throw new Error(JSON.stringify({operation,recorded,borrowed,i,x,y,actual:actual.slice(i*2,i*2+2),expected,error,tolerance,errors}));
              maxError=Math.max(maxError,error);
            }
            if(JSON.stringify(output.shape)!==JSON.stringify([1,count]))throw new Error('Pointwise output lost logical input shape.');
            if(output.precision!=='two-f32'||output.components!==2||output.elementStrideBytes!==8)throw new Error('Missing output pair metadata.');
            rows.push({operation,recorded,borrowed,maxError,actual});
          } finally { if(recorder)await recorder.abort(); }
        }
      }
      for(const recorded of [false,true]) {
        const a=pair(input),recorder=recorded?new CommandRecorder(device,'paired_inplace'):null;
        try {
          const output=recorded?await recordScale(recorder,a,scale,{count,inplace:true}):await runScale(a,scale,{count,inplace:true});
          if(JSON.stringify(output.shape)!==JSON.stringify([1,count]))throw new Error('In-place scale lost shape.');
          if(output.buffer!==a.buffer)throw new Error('In-place scale changed ownership.');
          if(recorder)await recorder.submitAndWait();
          const actual=Array.from(new Float32Array(await readBufferSlice(output.buffer,0,bytes)));
          for(let i=0;i<count;i++)if(Math.abs(actual[i*2]+actual[i*2+1]-(input[i*2]+input[i*2+1])*scale)>tolerance*1000)throw new Error('In-place value mismatch.');
          rows.push({operation:'scale_inplace',recorded,borrowed:true,actual});
        } finally {if(recorder)await recorder.abort();}
      }
      await device.queue.onSubmittedWorkDone();for(const buffer of pooled)releaseBuffer(buffer);pooled.clear();
      await getBufferPool().settleDeferredDestruction();
      const stats=getBufferPool().getStats();if(stats.activeBuffers!==0||errors.length)throw new Error(JSON.stringify({stats,errors}));
      return {rows,errors,stats,tolerance,input:Array.from(input),second:Array.from(second),scale,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture},settled:true};
    } finally {
      await device.queue.onSubmittedWorkDone();for(const buffer of pooled)releaseBuffer(buffer);for(const buffer of owned)buffer.destroy();destroyBufferPool();destroyDevice();
    }
  });
  receipt.browser = browser.version(); receipt.passed = receipt.result.rows.length === 26 && receipt.result.settled;
} catch (error) { receipt.error = error.stack; throw error; }
finally {
  await browser?.close(); server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
  await writeFile(destination, JSON.stringify(receipt, null, 2) + '\n');
}
console.log(JSON.stringify(receipt)); assert(receipt.passed, 'Paired pointwise wrappers failed.');
