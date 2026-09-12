/** CLI driver for the self-hosted compiler. Only this driver reads input files. */
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { assemble } from './assemble.mjs';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const args=process.argv.slice(2);
const targetIndex=args.indexOf('--target');
const target=targetIndex<0?'js':args[targetIndex+1];
const input=args[0];
if(!input || input.startsWith('--') || !['js','wgsl','wasm'].includes(target)) {
 console.error('Usage: node compiler/compile.mjs program.sprout|program.json [--target js|wgsl|wasm]');process.exit(1);
}
try {
 const text=fs.readFileSync(input,'utf8');
 const program=input.endsWith('.json')?JSON.parse(text):assemble(text);
 const context=vm.createContext({__QUINE_LIBRARY__:true,TextDecoder,TextEncoder});
 vm.runInContext(fs.readFileSync(path.join(root,'quines/05-compiler.js'),'utf8'),context,{timeout:5000});
 process.stdout.write(context.Quine.compile(program,target));
} catch(error) {console.error(error.message);process.exitCode=1;}
