import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const root=path.dirname(fileURLToPath(import.meta.url)), execute=promisify(execFile);
const teachers=JSON.parse(await fs.readFile(path.join(root,'teachers.json'))), out=path.join(root,'gates');
await fs.mkdir(out);
const receipt={schema:'doppler.amd-host-untimed-gates.v1',sessions:[],failures:[]};
for (const teacher of teachers) for (const lane of ['baseline','candidate']) {
  const id=teacher.id+'-'+lane, config=JSON.parse(await fs.readFile(path.join(root,'runs',teacher.id+'-0-'+lane+'-config.json')));
  config.run.browser.userDataDir=path.join(root,'gate-browser-'+lane);config.run.browser.port=19914;
  await fs.writeFile(path.join(out,id+'-config.json'),JSON.stringify(config,null,2));
  let stdout='',stderr='';console.log('START',id);
  try {
    ({stdout,stderr}=await execute(process.execPath,[path.join(root,'gate-runtime/src/cli/doppler-cli.js'),'bench','--config',path.join(out,id+'-config.json'),'--json'],{cwd:path.join(root,'gate-runtime'),timeout:240000,maxBuffer:96*1024*1024}));
    const data=JSON.parse(stdout);if(!data.ok)throw Error(JSON.stringify(data.error));receipt.sessions.push({id,prompt:teacher.id,lane,outputIds:data.result.metrics.referenceTranscript.tokens.ids});console.log('DONE',id);
  } catch(error) {stdout=error.stdout??stdout;stderr=error.stderr??stderr;receipt.failures.push({id,error:String(error.stack??error)});}
  finally {await fs.writeFile(path.join(out,id+'.json'),stdout);await fs.writeFile(path.join(out,id+'.stderr'),stderr);await fs.writeFile(path.join(out,'receipt.json'),JSON.stringify(receipt,null,2));}
}
receipt.complete=true;await fs.writeFile(path.join(out,'receipt.json'),JSON.stringify(receipt,null,2));
