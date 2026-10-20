// Verify provisioned remote bytes against the independently qualified successor.
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir,copyFile,stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const [runtime,source,destination,evidence]=process.argv.slice(2).map(value=>resolve(value));
const {hash}=await import(pathToFileURL(resolve(runtime,'src/storage/blake3.js')));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const expected=JSON.parse(await readFile(resolve(evidence,'successor-js-verification.json'),'utf8'));
const manifest=await readFile(resolve(evidence,'successor-manifest.json'));
assert.equal(sha(manifest),expected.manifestSha256);
assert.equal(sha(await readFile(resolve(source,'manifest.json'))),'83e5a01a21e234d238ea3ed5b0b234cc52781ddfce51861abd0fdf0ae3199028');
await mkdir(destination);const receipt={passed:false,source,destination,manifestSha256:sha(manifest),files:[],verification:'SHA256 and frozen standard-BLAKE3 JS against independently Python-verified M3 successor; no independent BLAKE3 implementation on this host'};
try{
 for(const row of expected.files){
  const path=resolve(source,row.file);const before=await stat(path);const bytes=await readFile(path);
  assert.equal(sha(bytes),row.sha256);assert.equal(Buffer.from(await hash(bytes)).toString('hex'),row.blake3);
  await copyFile(path,resolve(destination,row.file));const copy=await readFile(resolve(destination,row.file));
  assert.equal(sha(copy),row.sha256);assert.equal(Buffer.from(await hash(copy)).toString('hex'),row.blake3);
  assert.equal((await stat(path)).mtimeMs,before.mtimeMs);receipt.files.push({...row,bytes:bytes.length});
 }
 await writeFile(resolve(destination,'manifest.json'),manifest);receipt.passed=true;
}finally{await writeFile(resolve(evidence,'vulkan-successor-verification.json'),JSON.stringify(receipt,null,2)+'\n');}
console.log(JSON.stringify({passed:receipt.passed,files:receipt.files.length}));
