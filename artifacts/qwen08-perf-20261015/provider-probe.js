import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
const result={createdAt:new Date().toISOString(),providers:[]};
for(const specifier of ['webgpu','file:///Users/xyz/deco/doe/packages/doe-gpu/src/index.js']) {
  let gpu,device;
  try {
    const module=await import(specifier);gpu=module.create(['enable-dawn-features=allow_unsafe_apis']);
    const adapter=await gpu.requestAdapter();
    if(!adapter)throw new Error('No adapter');
    const record={specifier,info:{...adapter.info},features:[...adapter.features],wgslFeatures:[...(gpu.wgslLanguageFeatures??[])]};
    device=await adapter.requestDevice({requiredFeatures:['shader-f16','subgroups'].filter(x=>adapter.features.has(x))});
    record.deviceCreated=true;result.providers.push(record);
  }catch(error){result.providers.push({specifier,error:String(error),stack:error.stack});}
  finally{device?.destroy();gpu?.destroy?.();}
}
const lib=process.env.DOE_WEBGPU_LIB;
if(lib)result.doeLibrary={path:lib,sha256:createHash('sha256').update(await fs.readFile(lib)).digest('hex')};
await fs.writeFile(new URL('./provider-probe.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
