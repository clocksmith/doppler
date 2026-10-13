// CPU-only reuse of existing delivery/source-fact owners before physical opening.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, link, copyFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const root='/var/tmp/reploid-gate-three-stage-20261013';
const model='/var/tmp/doppler-gate-model-20261013';
const previous='/var/tmp/doppler-sigmoid-model-20261013';
const config=JSON.parse(await readFile(root+'/config-preparation.json'));
const {verifyNumericalTestDelivery}=await import(root+'/tests/fixtures/numerical-test-delivery.js');
const {bindPairedPartitionFixture}=await import(root+'/tests/fixtures/paired-partition-binding.js');
const delivery=await verifyNumericalTestDelivery({root,bindingPath:config.delivery.path,bindingSha256:config.delivery.hash.slice(7),archivePath:config.archive.path,runtimeDirectory:config.runtimeDirectory});
const manifestBytes=await readFile(config.manifest.path), manifest=JSON.parse(manifestBytes);
const bound=await bindPairedPartitionFixture({bindingBytes:await readFile(config.pairedBinding.path),graphBytes:await readFile(config.graph.path),modelBytes:manifestBytes,archiveBytes:await readFile(config.archive.path),referenceIdentity:JSON.parse(await readFile(config.reference.path)).modelIdentity});
await mkdir(model);
await copyFile(config.manifest.path,model+'/manifest.json');
const {hash:blake3}=await import(config.runtimeDirectory+'/src/storage/blake3.js');
const shards=[];
for(const shard of manifest.shards){
 const bytes=await readFile(previous+'/'+shard.filename);assert.equal(bytes.length,shard.size);assert.equal(Buffer.from(await blake3(bytes)).toString('hex'),shard.blake3);
 await link(previous+'/'+shard.filename,model+'/'+shard.filename);shards.push({filename:shard.filename,bytes:bytes.length,blake3:shard.blake3});
}
const tokenizer=await readFile(previous+'/tokenizer.json');assert.equal(createHash('sha256').update(tokenizer).digest('hex'),'8fc3b6de02de5a8e21d3867aba335e2d9a3c2263201f55daaed1feab3541bea4');
await link(previous+'/tokenizer.json',model+'/tokenizer.json');
await writeFile(model+'/verification.json',JSON.stringify({delivery,bound,shards,tokenizerSha256:'8fc3b6de02de5a8e21d3867aba335e2d9a3c2263201f55daaed1feab3541bea4',model,node:process.version,gpuOpened:false},null,2)+'\n');
console.log(JSON.stringify({delivery,shards:shards.length,gpuOpened:false}));
