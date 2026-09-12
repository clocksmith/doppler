import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { assemble } from '../compiler/assemble.mjs';
import { interpret } from './reference.mjs';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const evidence = path.join(root, 'verification/sprout-three-backend-2026-09-12');
const read = relative => fs.readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const checks = [];
const sameBytes = (left, right) => assert.deepEqual(Buffer.from(left), Buffer.from(right));
const load = source => {
  const context = vm.createContext({ __QUINE_LIBRARY__: true, TextEncoder, TextDecoder });
  vm.runInContext(source, context, { timeout: 5000 });
  return context.Quine;
};
const check = async (name, work) => {
  await work();
  checks.push({ name, status: 'passed' });
};
fs.mkdirSync(evidence, { recursive: true });
const report = { schema: 'sprout-three-backend-cpu-v1', started: new Date().toISOString(),
  runtime: process.version, platform: `${process.platform}/${process.arch}`, checks };

try {
  const source = read('quines/05-compiler.js').toString('utf8');
  const shader = read('quines/05-compiler.wgsl').toString('utf8');
  const binary = read('quines/05-compiler.wasm');
  const seed = load(source);
  await check('Bootstrap agrees with the executing Sprout JS compiler for all three backends', () => {
    assert.equal(seed.self(), source);
    assert.equal(seed.wgsl(), shader);
    sameBytes(seed.wasm(), binary);
  });
  await check('Native Wasm validates and has no imports', () => {
    assert.equal(WebAssembly.validate(binary), true);
    assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(binary)), []);
  });
  const descendant = await seed.fromWasm(binary);
  await check('Actual Wasm descendant emits byte-identical JS, WGSL, and Wasm', () => {
    assert.equal(descendant.js(), source);
    assert.equal(descendant.wgsl(), shader);
    sameBytes(descendant.self(), binary);
    assert.equal(load(descendant.js()).self(), source);
  });
  await check('Independent Sprout interpreter emits the same binary', () => {
    const result = interpret(seed.program, seed.program, 2);
    sameBytes(Uint8Array.from(result.text, character => character.charCodeAt(0)), binary);
  });
  await check('Compiler assembly describes the embedded program exactly', () => {
    assert.deepEqual(Array.from(seed.program), Array.from(assemble(read('compiler/compiler.sprout').toString('utf8'))));
  });

  const examples = JSON.parse(read('compiler/examples.json'));
  for (const { name, expected } of examples) {
    const program = JSON.parse(read(`compiler/${name}.json`));
    for (const [runtime, compiler] of [['js', seed], ['wasm', descendant]]) {
      for (const target of ['js', 'wgsl', 'wasm']) {
        await check(`${runtime} compiles unrelated ${name} to ${target}`, async () => {
          const output = compiler.compile(program, target);
          sameBytes(typeof output === 'string' ? Buffer.from(output) : output, read(`compiler/${name}.${target}`));
          if (target === 'js') assert.equal(load(output).execute(), expected);
          if (target === 'wasm') assert.equal((await seed.fromWasm(output)).js(), expected);
        });
      }
    }
  }

  const cases = [
    ['unsigned arithmetic', '4294967295,0,2147483647,1,0,0,1', `
      set 2 4294967295
      decimal 2
      set 6 44
      put 6
      set 3 65536
      mul 4 3 3
      decimal 4
      put 6
      set 3 2
      div 4 2 3
      decimal 4
      put 6
      mod 4 2 3
      decimal 4
      put 6
      set 3 0
      div 4 2 3
      decimal 4
      put 6
      mod 4 2 3
      decimal 4
      put 6
      lt 4 3 2
      decimal 4`],
    ['nested structured control', 'AAAAAAZ', `
      set 2 2
      set 3 1
      while 2
        set 4 3
        while 4
          if 4
            set 5 65
          else
            set 5 66
          fi
          put 5
          sub 4 4 3
        end
        sub 2 2 3
      end
      if 2
        set 5 88
      else
        set 5 90
      fi
      put 5`],
    ['bounds and register 63', '213310', `
      set 63 0
      input 2 63
      decimal 2
      set 63 4294967295
      data 2 63
      decimal 2`],
    ['zero output is valid', '', 'nop'],
  ];
  for (const [name, expected, body] of cases) {
    const program = assemble(`.data\n.code\n${body}\n`);
    await check(name, async () => {
      assert.equal(interpret(program).text, expected);
      assert.equal(load(seed.compile(program, 'js')).execute(), expected);
      assert.equal((await seed.fromWasm(descendant.compile(program, 'wasm'))).js(), expected);
    });
  }
  await check('put preserves every boundary byte, not just ASCII', async () => {
    const program = assemble('.data\n.code\nset 2 0\nput 2\nset 2 127\nput 2\nset 2 128\nput 2\nset 2 255\nput 2\n');
    sameBytes(load(seed.compile(program)).executeBytes(), [0, 127, 128, 255]);
    sameBytes((await seed.fromWasm(descendant.compile(program, 'wasm'))).compile(null, 'wasm'), [0, 127, 128, 255]);
  });
  const malformed = [
    [], [0, 0, 0], [21331, 0, 1, 0], [21331, 1, 0],
    [21331, 0, 4, 19, 0, 0, 0], [21331, 0, 4, 1, 64, 0, 0],
    [21331, 0, 4, 2, 0, 64, 0], [21331, 0, 4, 3, 0, 0, 64],
    [21331, 0, 4, 14, 0, 0, 0], [21331, 0, 4, 17, 0, 0, 0],
    [21331, 0, 8, 16, 0, 0, 0, 15, 0, 0, 0],
    [21331, 0, 16, 16, 0, 0, 0, 17, 0, 0, 0, 17, 0, 0, 0, 18, 0, 0, 0],
  ];
  const deep = Array.from({ length: 65 }, () => [16, 2, 0, 0]).flat();
  deep.push(...Array.from({ length: 65 }, () => [18, 0, 0, 0]).flat());
  malformed.push([21331, 0, deep.length, ...deep]);
  for (const [index, program] of malformed.entries()) {
    await check(`Malformed program ${index} is rejected by JS and native Wasm`, () => {
      assert.throws(() => seed.compile(program, 'wasm'));
      assert.throws(() => descendant.compile(program, 'wasm'));
    });
  }
  await check('Raw native ABI rejects invalid targets and input pointers', async () => {
    const { instance } = await WebAssembly.instantiate(binary);
    assert.throws(() => instance.exports.run(3, 0, 0), WebAssembly.RuntimeError);
    assert.throws(() => instance.exports.run(0, 524292, 3), WebAssembly.RuntimeError);
  });
  report.status = 'passed';
  report.sources = Object.fromEntries(['js', 'wgsl', 'wasm'].map(extension => {
    const bytes = read(`quines/05-compiler.${extension}`);
    return [extension, { bytes: bytes.length, sha256: hash(bytes) }];
  }));
  console.log(`${checks.length} Sprout Wasm checks passed. No GPU claims in this CPU receipt.`);
} catch (error) {
  report.status = 'failed';
  report.error = error.stack;
  console.error(error);
  process.exitCode = 1;
} finally {
  report.finished = new Date().toISOString();
  fs.writeFileSync(path.join(evidence, 'cpu.json'), JSON.stringify(report, null, 2) + '\n');
}
