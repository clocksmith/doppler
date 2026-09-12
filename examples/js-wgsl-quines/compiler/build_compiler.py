"""Bootstrap a self-hosting Sprout -> JavaScript/WGSL compiler.

Executed by ../build.py. The generated compiler never imports this builder.
Sprout is structured register assembly. Compiled outputs contain direct host
statements; they do not interpret Sprout at runtime.
"""
OPS = ['nop','set','mov','add','sub','mul','div','mod','eq','lt','data','input','put','decimal','while','end','if','else','fi']
OPNUM={name:i for i,name in enumerate(OPS)}
JS_OP=[
';\n','r[~a]=~b;\n','r[~a]=r[~b];\n','r[~a]=(r[~b]+r[~c])>>>0;\n',
'r[~a]=(r[~b]-r[~c])>>>0;\n','r[~a]=Math.imul(r[~b],r[~c])>>>0;\n',
'r[~a]=Math.floor(r[~b]/r[~c])>>>0;\n','r[~a]=(r[~b]%r[~c])>>>0;\n',
'r[~a]=Number(r[~b]===r[~c]);\n','r[~a]=Number(r[~b]<r[~c]);\n',
'r[~a]=at(P,r[~b]);\n','r[~a]=at(input,r[~b]);\n','put(r[~a]);\n','decimal(r[~a]);\n',
'while(r[~a]!==0){\n','}\n','if(r[~a]!==0){\n','}else{\n','}\n']
WG_OP=[
'{}\n','r[~a]=~bu;\n','r[~a]=r[~b];\n','r[~a]=r[~b]+r[~c];\n',
'r[~a]=r[~b]-r[~c];\n','r[~a]=r[~b]*r[~c];\n',
'r[~a]=safe_div(r[~b],r[~c]);\n','r[~a]=safe_mod(r[~b],r[~c]);\n',
'r[~a]=select(0u,1u,r[~b]==r[~c]);\n','r[~a]=select(0u,1u,r[~b]<r[~c]);\n',
'r[~a]=data_at(r[~b]);\n','r[~a]=input_at(r[~b]);\n','put(r[~a]);\n','decimal(r[~a]);\n',
'while(r[~a]!=0u){\n','}\n','if(r[~a]!=0u){\n','}else{\n','}\n']

JS_VALIDATE=r'''function at(a, i) { return i < a.length ? a[i] : 0; }
function validateProgram(p) {
  if (!p || p.length < 3 || p.length > 60000 || p[0] !== 21331 || p[2] % 4 || p.length !== 3+p[1]+p[2]) throw new Error("Invalid Sprout header.");
  if (Array.from(p).some(n => !Number.isSafeInteger(n) || n < 0 || n > 4294967295)) throw new Error("Sprout words must be u32.");
  const stack = [];
  for (let pc = 3+p[1]; pc < p.length; pc += 4) {
    const [op,a,b,c] = Array.from(p.slice(pc,pc+4));
    if (op > 18 || a >= 64 || (op >= 2 && op <= 11 && b >= 64) || (op >= 3 && op <= 9 && c >= 64)) throw new Error("Invalid Sprout instruction.");
    if (op === 14 || op === 16) stack.push(op);
    if (op === 15 && stack.pop() !== 14) throw new Error("Unbalanced while.");
    if (op === 17) { if (stack.pop() !== 16) throw new Error("Unbalanced else."); stack.push(17); }
    if (op === 18 && ![16,17].includes(stack.pop())) throw new Error("Unbalanced if.");
    if (stack.length > 64) throw new Error("Control nesting is too deep.");
  }
  if (stack.length) throw new Error("Unclosed control block.");
}'''

JS_PARTS=[
'(() => {\n"use strict";\nconst PROGRAM_LENGTH = ',
';\nconst P = Object.freeze([',
']);\n'+JS_VALIDATE+r'''
function execute(input = P, target = 0) {
  validateProgram(input);
  if (target !== 0 && target !== 1) throw new Error("Use target 0 (JS) or 1 (WGSL).");
  const r = new Uint32Array(64);
  r[0] = target; r[1] = input.length;
  const output = [];
  const put = c => { if (c > 127 || output.length >= 262143) throw new Error("Invalid or oversized compiler output."); output.push(c); };
  const decimal = n => { for (const c of String(n >>> 0)) put(c.charCodeAt(0)); };
''',
r'''  return new TextDecoder().decode(Uint8Array.from(output));
}
'''+HOST+PRINT_JS+r'''
const api = { id: "compiler", program: P, execute, self: () => execute(P, 0), wgsl: () => execute(P, 1),
  compile: (program, target = "js") => execute(program, target === "js" ? 0 : target === "wgsl" ? 1 : -1),
  run: (target = "js", input = null) => {
    if (!["js", "wgsl"].includes(target)) throw new Error("Use js or wgsl.");
    if (input !== null) validateProgram(input);
    return withDevice(device => dispatchText(device, execute(P, 1), "main", { TARGET: target === "js" ? 0 : 1, EXTERNAL: input === null ? 0 : 1 }, input ?? []));
  }
};
const DEFAULT_GPU = false;
'''+AUTORUN+'\n})();\n'
]

WG_RUNTIME=r'''override TARGET: u32 = 1u;
override EXTERNAL: u32 = 0u;
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<storage, read> supplied: array<u32>;
var<private> cursor: u32 = 1u;
var<private> bad: bool = false;
fn put(c: u32) {
  if (c > 127u || cursor >= arrayLength(&output)) { bad = true; return; }
  output[cursor] = c; cursor += 1u;
}
fn decimal(n: u32) {
  var divisor = 1u;
  while (n / divisor >= 10u) { divisor *= 10u; }
  var value = n;
  loop {
    put(48u + value / divisor); value %= divisor;
    if (divisor == 1u) { break; }
    divisor /= 10u;
  }
}
fn data_at(i: u32) -> u32 { if (i >= PROGRAM_LENGTH) { return 0u; } return P[i]; }
fn input_length() -> u32 { if (EXTERNAL == 0u) { return PROGRAM_LENGTH; } return supplied[0]; }
fn input_at(i: u32) -> u32 {
  if (EXTERNAL == 0u) { return data_at(i); }
  if (i >= supplied[0] || i + 1u >= arrayLength(&supplied)) { return 0u; }
  return supplied[i + 1u];
}
fn safe_div(a: u32, b: u32) -> u32 { if (b == 0u) { return 0u; } return a / b; }
fn safe_mod(a: u32, b: u32) -> u32 { if (b == 0u) { return 0u; } return a % b; }
fn valid_input() -> bool {
  let n = input_length();
  if (TARGET > 1u || n < 3u || n > 60000u || input_at(0u) != 21331u) { return false; }
  if (EXTERNAL != 0u && n >= arrayLength(&supplied)) { return false; }
  let data_length = input_at(1u); let code_length = input_at(2u);
  if (data_length > n || code_length > n || code_length % 4u != 0u || 3u + data_length + code_length != n) { return false; }
  var stack: array<u32, 64>;
  var depth = 0u;
  for (var pc = 3u + data_length; pc < n; pc += 4u) {
    let op = input_at(pc); let a = input_at(pc+1u); let b = input_at(pc+2u); let c = input_at(pc+3u);
    if (op > 18u || a >= 64u || (op >= 2u && op <= 11u && b >= 64u) || (op >= 3u && op <= 9u && c >= 64u)) { return false; }
    if (op == 14u || op == 16u) {
      if (depth == 64u) { return false; }
      stack[depth] = op; depth += 1u;
    } else if (op == 15u) {
      if (depth == 0u) { return false; } depth -= 1u;
      if (stack[depth] != 14u) { return false; }
    } else if (op == 17u) {
      if (depth == 0u || stack[depth-1u] != 16u) { return false; }
      stack[depth-1u] = 17u;
    } else if (op == 18u) {
      if (depth == 0u) { return false; } depth -= 1u;
      if (stack[depth] != 16u && stack[depth] != 17u) { return false; }
    }
  }
  return depth == 0u;
}
@compute @workgroup_size(1)
fn main() {
  if (!valid_input()) { output[0] = 0u; return; }
  var r: array<u32, 64>;
  r[0] = TARGET; r[1] = input_length();
'''
WG_PARTS=['const PROGRAM_LENGTH = ', 'u;\nconst P = array<u32, PROGRAM_LENGTH>(', ');\n'+WG_RUNTIME, '  output[0] = select(cursor - 1u, 0u, bad);\n}\n']

# The compiler's own embedded data consists only of generic code-emission
# templates. It also includes its Sprout instructions, compiled below.
# Keep the small language; add native binary emission in the backend builder.
AUTORUN = AUTORUN.replace('const task =', 'const task=')
exec((ROOT / 'compiler/build_wasm.py').read_text(), globals())
