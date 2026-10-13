import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
const root=process.cwd();
const server=createServer(async(req,res)=>{try{
 const pathname=new URL(req.url,'http://localhost').pathname;
 const suffix=pathname.startsWith('/config/')?'self'+pathname:pathname.startsWith('/vendor/')?pathname.slice(1):null;
 if(!suffix||!['GET','HEAD'].includes(req.method))throw Error('route');
 const file=resolve(root,suffix);if(!file.startsWith(root+sep))throw Error('path');
 const bytes=await readFile(file);res.setHeader('Content-Type',extname(file)==='.js'?'text/javascript':extname(file)==='.json'?'application/json':extname(file)==='.wasm'?'application/wasm':'application/octet-stream');res.end(req.method==='HEAD'?undefined:bytes);
}catch{res.writeHead(404).end();}});
await new Promise(resolve=>server.listen(29850,'127.0.0.1',resolve));
process.env.REPLOID_MODEL_PORT='29851';const model=await import('./tests/fixtures/numerical-model-server.js');
console.log(JSON.stringify({app:'http://127.0.0.1:29850',model:'http://127.0.0.1:29851',pid:process.pid}));
process.on('SIGTERM',()=>{server.closeAllConnections();server.close();model.server.closeAllConnections();model.server.close();});
