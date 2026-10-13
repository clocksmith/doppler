// Compare existing sampling failures with the pre-repair executor, without changing files.
import { execFileSync } from 'node:child_process';
import { registerHooks } from 'node:module';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
const allowed = new Set(['tests/kernels/capsule-token-selection-physical.test.js','tests/kernels/sampling-distribution-physical.test.js']);
const file=process.argv[2];if(!allowed.has(file))throw Error('Unadmitted baseline test');
const source=execFileSync('git',['show','d777f2a5:src/gpu/kernels/kernel-execution.js'],{encoding:'utf8'});
const url=pathToFileURL(resolve('src/gpu/kernels/kernel-execution.js')).href;
console.log(JSON.stringify({scope:'Pre-repair executor, same sampling test and shader bytes; not model qualification',
 sourceCommit:'d777f2a5',file,executorSha256:createHash('sha256').update(source).digest('hex')}));
const hooks=registerHooks({load(target,context,next){return target===url ? {format:'module',source,shortCircuit:true}:next(target,context);}});
try { await import(pathToFileURL(resolve(file))); } finally { hooks.deregister(); }
