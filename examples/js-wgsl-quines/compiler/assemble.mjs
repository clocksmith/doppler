import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
const names = ['nop','set','mov','add','sub','mul','div','mod','eq','lt','data','input','put','decimal','while','end','if','else','fi'];
export function assemble(text) {
  const data = [], code = []; let section = '';
  for (const [index, raw] of text.split(/\r?\n/).entries()) {
    const line = raw.split(';')[0].trim(); if (!line) continue;
    if (line === '.data' || line === '.code') { section = line; continue; }
    const parts = line.split(/\s+/);
    if (section === '.data') data.push(...parts.map(Number));
    else if (section === '.code') {
      const op = names.indexOf(parts.shift());
      if (op < 0 || parts.length > 3) throw new Error(`Invalid instruction at line ${index+1}.`);
      const args = parts.map(Number); while(args.length < 3) args.push(0);
      code.push(op, ...args);
    } else throw new Error(`Expected .data or .code before line ${index+1}.`);
  }
  const p = [21331, data.length, code.length, ...data, ...code];
  if (p.some(n => !Number.isSafeInteger(n) || n < 0 || n > 4294967295)) throw new Error('All words must be unsigned 32-bit integers.');
  return p;
}
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('Usage: node compiler/assemble.mjs program.sprout');
  process.stdout.write(JSON.stringify(assemble(fs.readFileSync(process.argv[2], 'utf8'))) + '\n');
}
