import { readFile, writeFile } from 'node:fs/promises';
import { bootstrapNativeTrainingHost, releaseNativeTrainingHost, loadNativeQwenTrainingPipeline, createNativeQwenLoRATrainer } from 'doppler-gpu/training';
const root='/Users/xyz/deco/doppler/reports/community-lora-20261017';
const read=async (name)=>JSON.parse(await readFile(root+'/'+name,'utf8'));
const config=await read('apply.json'), round=(await read('round.json')).result;
await bootstrapNativeTrainingHost(); let loaded, trainer;
try {
 loaded=await loadNativeQwenTrainingPipeline(config.modelUrl);
 trainer=createNativeQwenLoRATrainer({...config.trainer,pipeline:loaded.pipeline,baseModelId:loaded.manifest.modelId,layerIdx:loaded.pipeline.modelConfig.numLayers-1,module:'down_proj'});
 trainer.restoreCheckpoint(round.checkpoint);
 const contributions=await Promise.all(['contribution-m3.json','contribution-amd.json'].map(async name=>(await read(name)).result));
 const applied=await trainer.applyGradientContributions(round,contributions);
 const adapter=await trainer.exportAdapter(config.adapter);
 const expected=(await read('applied.json')).result.weightsSha256;
 if(adapter.weightsSha256!==expected) throw new Error('Packed application update differs');
 const receipt={passed:true,entry:'doppler-gpu/training',stepCount:applied.stepCount,contributors:applied.contributors,weightsSha256:adapter.weightsSha256,version:adapter.manifest.version};
 await writeFile(root+'/package/application.json',JSON.stringify(receipt,null,2)+'\n');
 console.log(JSON.stringify(receipt));
} finally { trainer?.dispose(); await loaded?.pipeline?.unload(); await releaseNativeTrainingHost(); }
