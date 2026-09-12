#!/usr/bin/env python3
"""Reproducible construction of five quines. No generated quine reads this file.

The first four use a conventional fixed point: template bytes + an emitter that
substitutes a numeric representation of those same template bytes. Substitution
also happens in WGSL, not in a hidden host callback.
"""
from pathlib import Path
import json
import hashlib
import re

ROOT = Path(__file__).resolve().parent
HOST = (ROOT / 'lib/webgpu.js').read_text()

def jsstr(s):
    return json.dumps(s, ensure_ascii=True, separators=(',', ':'))

def numbers(xs):
    return ','.join(str(x) for x in xs)

EXPAND_JS = r'''function expand(start = 0, end = Q.length) {
  let result = "";
  for (let i = start; i < end; i++) {
    const c = Q[i];
    if (c !== 126) { result += String.fromCharCode(c); continue; }
    const tag = Q[++i];
    if (tag === 68) result += Q.join(",");
    else if (tag === 78) result += Q.length;
    else if (tag === 74) result += CUT;
    else throw new Error("Unknown template instruction.");
  }
  return result;
}'''

EMITTER_WGSL = r'''@group(0) @binding(0) var<storage, read_write> output: array<u32>;
var<private> cursor: u32 = 1u;
fn put(c: u32) {
  if (cursor < arrayLength(&output)) { output[cursor] = c; }
  cursor += 1u;
}
fn decimal(n: u32) {
  var divisor = 1u;
  while (n / divisor >= 10u) { divisor *= 10u; }
  var value = n;
  loop {
    put(48u + value / divisor);
    value %= divisor;
    if (divisor == 1u) { break; }
    divisor /= 10u;
  }
}
fn data_literal() {
  for (var i = 0u; i < COUNT; i += 1u) {
    if (i != 0u) { put(44u); }
    decimal(Q[i]);
  }
}
fn expand(first: u32, last: u32) {
  var i = first;
  while (i < last) {
    let c = Q[i];
    if (c != 126u) { put(c); }
    else {
      i += 1u;
      let tag = Q[i];
      if (tag == 68u) { data_literal(); }
      else if (tag == 78u) { decimal(COUNT); }
      else if (tag == 74u) { decimal(CUT); }
    }
    i += 1u;
  }
}
'''

PRINT_JS = r'''function printSource(text) {
  if (typeof process !== "undefined" && process.stdout?.write) process.stdout.write(text);
  else if (typeof document !== "undefined") {
    const area = document.createElement("textarea");
    area.readOnly = true; area.value = text;
    area.style.cssText = "width:95vw;height:70vh;font:12px monospace";
    document.body.append(area);
  } else console.log(text);
}'''

AUTORUN = r'''globalThis.Quine = Object.freeze(api);
if (!globalThis.__QUINE_LIBRARY__) {
  const args = typeof process !== "undefined" ? process.argv.slice(2) : [];
  const task = args.includes("--wgsl") ? Promise.resolve({ text: api.wgsl() })
    : args.includes("--cpu") || (!DEFAULT_GPU && !args.includes("--gpu"))
      ? Promise.resolve({ text: api.self() }) : api.run();
  globalThis.QuineDone = task.then(result => { printSource(result.text); return result; });
  globalThis.QuineDone.catch(error => {
    if (typeof process !== "undefined") { console.error(error.message); process.exitCode = 1; }
    else { const pre = document.createElement("pre"); pre.textContent = error.message; document.body.append(pre); }
  });
}'''

def expand_python(q, cut=0, first=0, last=None):
    last = len(q) if last is None else last
    out=[]; i=first
    while i<last:
        c=q[i]
        if c==126:
            i+=1
            out.append({68:numbers(q),78:str(len(q)),74:str(cut)}[q[i]])
        else: out.append(chr(c))
        i+=1
    return ''.join(out)

def make_shader_js(extra='', entries='@compute @workgroup_size(1) fn main() { expand(0u, COUNT); output[0] = cursor - 1u; }\n'):
    s = 'const COUNT = @N@u;\nconst CUT = @J@u;\nconst Q = array<u32, @N@>(@D@);\n' + EMITTER_WGSL + extra + entries
    return 'function shader() {\n  return '+jsstr(s)+'.replaceAll("@N@", String(Q.length)).replaceAll("@J@", String(CUT)).replaceAll("@D@", Q.join(","));\n}'

def assemble_quine(name, template, cut=0):
    q=list(template.encode('ascii'))
    text=expand_python(q,cut)
    path=ROOT/'quines'/name; path.write_text(text)
    return text,q

full = '\n'.join([
    '(() => {', '"use strict";', '// A complete application: its shader reconstructs this entire JavaScript file.',
    'const Q = Object.freeze([~D]);', 'const CUT = 0;', EXPAND_JS,
    make_shader_js(), HOST, PRINT_JS,
    'const api = { id: "application", data: Q, self: () => expand(), wgsl: shader,',
    '  run: () => withDevice(device => dispatchText(device, shader())) };',
    'const DEFAULT_GPU = true;', AUTORUN, '})();', ''
])
full_src, full_q=assemble_quine('01-application.js',full)

# Nested-comment language separation. The entire WGSL-active suffix occupies
# ONE physical line, and is consequently a JavaScript line comment.
poly_w='const COUNT = ~Nu; const CUT = 0u; const Q = array<u32, ~N>(~D); '
poly_w+=' '.join(EMITTER_WGSL.split())
poly_w+=' @compute @workgroup_size(1) fn main() { expand(0u, COUNT); output[0] = cursor - 1u; }'
poly='\n'.join([
    '/* /* */','(() => {','"use strict";',
    'const Q = Object.freeze([~D]);','const CUT = 0;',EXPAND_JS,HOST,PRINT_JS,
    'const api = { id: "polyglot", data: Q, self: () => expand(), wgsl: () => expand(),',
    '  run: () => withDevice(device => dispatchText(device, expand())) };',
    'const DEFAULT_GPU = false;',AUTORUN,'})();','// */'+poly_w,''
])
# Prevent accidental comment terminators in the JS-only region.
assert '/*' not in poly.split('\n',1)[1].split('\n// */')[0]
assert '*/' not in poly.split('\n',1)[1].split('\n// */')[0]
poly_src, poly_q=assemble_quine('02-polyglot.js',poly)
(ROOT/'quines/02-polyglot.wgsl').write_text(poly_src)

multi_j='\n'.join([
    '(() => {','"use strict";', '// Either language can emit itself or its partner.',
    'const Q = Object.freeze([~D]);','const CUT = ~J;',EXPAND_JS,HOST,PRINT_JS,
    'const api = { id: "multiquine", data: Q, cut: CUT, self: () => expand(0, CUT), wgsl: () => expand(CUT),',
    '  emit: target => target === "js" ? expand(0, CUT) : target === "wgsl" ? expand(CUT) : (() => { throw new Error("Use js or wgsl."); })(),',
    '  run: (target = "js") => withDevice(device => dispatchText(device, expand(CUT), target === "wgsl" ? "main" : "javascript")) };',
    'const DEFAULT_GPU = false;',AUTORUN,'})();',''
])
multi_w='const COUNT = ~Nu;\nconst CUT = ~Ju;\nconst Q = array<u32, ~N>(~D);\n'+EMITTER_WGSL
multi_w+='@compute @workgroup_size(1) fn main() { expand(CUT, COUNT); output[0] = cursor - 1u; }\n'
multi_w+='@compute @workgroup_size(1) fn javascript() { expand(0u, CUT); output[0] = cursor - 1u; }\n'
multi_q=list((multi_j+multi_w).encode('ascii'))
multi_src=expand_python(multi_q,len(multi_j),0,len(multi_j))
multi_shader=expand_python(multi_q,len(multi_j),len(multi_j))
(ROOT/'quines/03-multiquine.js').write_text(multi_src)
(ROOT/'quines/03-multiquine.wgsl').write_text(multi_shader)

# Hand-drawn 5x7 glyphs, not a redistributed font file. Unassigned characters
# are visible boxes; all characters also have lossless pixel metadata.
GLYPHS={
' ':['00000']*7,
'0':['01110','10001','10011','10101','11001','10001','01110'],
'1':['00100','01100','00100','00100','00100','00100','01110'],
'2':['01110','10001','00001','00010','00100','01000','11111'],
'3':['11110','00001','00001','01110','00001','00001','11110'],
'4':['00010','00110','01010','10010','11111','00010','00010'],
'5':['11111','10000','10000','11110','00001','00001','11110'],
'6':['01110','10000','10000','11110','10001','10001','01110'],
'7':['11111','00001','00010','00100','01000','01000','01000'],
'8':['01110','10001','10001','01110','10001','10001','01110'],
'9':['01110','10001','10001','01111','00001','00001','01110'],
'A':['01110','10001','10001','11111','10001','10001','10001'],
'B':['11110','10001','10001','11110','10001','10001','11110'],
'C':['01111','10000','10000','10000','10000','10000','01111'],
'D':['11110','10001','10001','10001','10001','10001','11110'],
'E':['11111','10000','10000','11110','10000','10000','11111'],
'F':['11111','10000','10000','11110','10000','10000','10000'],
'G':['01111','10000','10000','10111','10001','10001','01111'],
'H':['10001','10001','10001','11111','10001','10001','10001'],
'I':['01110','00100','00100','00100','00100','00100','01110'],
'J':['00111','00010','00010','00010','10010','10010','01100'],
'K':['10001','10010','10100','11000','10100','10010','10001'],
'L':['10000','10000','10000','10000','10000','10000','11111'],
'M':['10001','11011','10101','10101','10001','10001','10001'],
'N':['10001','11001','10101','10011','10001','10001','10001'],
'O':['01110','10001','10001','10001','10001','10001','01110'],
'P':['11110','10001','10001','11110','10000','10000','10000'],
'Q':['01110','10001','10001','10001','10101','10010','01101'],
'R':['11110','10001','10001','11110','10100','10010','10001'],
'S':['01111','10000','10000','01110','00001','00001','11110'],
'T':['11111','00100','00100','00100','00100','00100','00100'],
'U':['10001','10001','10001','10001','10001','10001','01110'],
'V':['10001','10001','10001','10001','10001','01010','00100'],
'W':['10001','10001','10001','10101','10101','10101','01010'],
'X':['10001','10001','01010','00100','01010','10001','10001'],
'Y':['10001','10001','01010','00100','00100','00100','00100'],
'Z':['11111','00001','00010','00100','01000','10000','11111'],
'.':['00000','00000','00000','00000','00000','00110','00110'],
',':['00000','00000','00000','00000','00110','00110','00100'],
';':['00000','00110','00110','00000','00110','00110','00100'],
':':['00000','00110','00110','00000','00110','00110','00000'],
'"':['01010','01010','00000','00000','00000','00000','00000'],
"'":['00100','00100','00000','00000','00000','00000','00000'],
'`':['01000','00100','00000','00000','00000','00000','00000'],
'=':['00000','00000','11111','00000','11111','00000','00000'],
'+':['00000','00100','00100','11111','00100','00100','00000'],
'-':['00000','00000','00000','11111','00000','00000','00000'],
'*':['00000','10101','01110','11111','01110','10101','00000'],
'/':['00001','00010','00010','00100','01000','01000','10000'],
'\\':['10000','01000','01000','00100','00010','00010','00001'],
'(' :['00010','00100','01000','01000','01000','00100','00010'],
')' :['01000','00100','00010','00010','00010','00100','01000'],
'[' :['01110','01000','01000','01000','01000','01000','01110'],
']' :['01110','00010','00010','00010','00010','00010','01110'],
'{' :['00011','00100','00100','11000','00100','00100','00011'],
'}' :['11000','00100','00100','00011','00100','00100','11000'],
'<' :['00010','00100','01000','10000','01000','00100','00010'],
'>' :['01000','00100','00010','00001','00010','00100','01000'],
'!' :['00100','00100','00100','00100','00100','00000','00100'],
'?' :['01110','10001','00001','00010','00100','00000','00100'],
'|' :['00100','00100','00100','00100','00100','00100','00100'],
'_' :['00000','00000','00000','00000','00000','00000','11111'],
'&' :['01100','10010','10100','01000','10101','10010','01101'],
'%' :['11001','11010','00100','01000','10110','00110','00000'],
'$' :['00100','01111','10100','01110','00101','11110','00100'],
'#' :['01010','11111','01010','01010','11111','01010','00000'],
'@' :['01110','10001','10111','10101','10111','10000','01111'],
'^' :['00100','01010','10001','00000','00000','00000','00000'],
'~' :['00000','00000','01001','10110','00000','00000','00000'],
'\n':['00000','00001','00101','01111','00100','00000','00000']
}
for c in 'abcdefghijklmnopqrstuvwxyz':
    upper=GLYPHS[c.upper()]
    # A compact small-cap treatment; byte identity is preserved in pixel metadata.
    GLYPHS[c]=['00000']+upper[:6]
font=[]
for code in range(128):
    rows=GLYPHS.get(chr(code),['11111','10001','10101','10101','10101','10001','11111'])
    font.extend(int(row,2) for row in rows)
paint = r'''@group(0) @binding(1) var<storage, read_write> pixels: array<u32>;
const FONT = array<u32, 896>(@FONT@);
@compute @workgroup_size(8, 8)
fn paint(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = 1024u;
  let height = ((output[0] + 128u) / 128u) * 10u;
  if (gid.x >= width || gid.y >= height) { return; }
  let column = gid.x / 8u;
  let row = gid.y / 10u;
  let cell = row * 128u + column;
  let x = gid.x % 8u;
  let y = gid.y % 10u;
  var c = 32u;
  if (cell == 0u) { c = 35u; }
  else if (cell <= output[0]) { c = output[cell]; }
  var color = 0xff19130fu;
  if (x >= 1u && x <= 5u && y >= 1u && y <= 7u) {
    let bit = (FONT[c * 7u + y - 1u] >> (5u - x)) & 1u;
    if (bit == 1u) { color = select(0xffdfd6c9u, 0xff95eac3u, c >= 48u && c <= 57u); }
  }
  if (x == 7u && y == 9u) {
    if (cell == 0u) { color = 0xff000000u | output[0]; }
    else { color = 0xff5aa500u | c; }
  }
  pixels[gid.y * width + gid.x] = color;
}
'''
visualrun = r'''async function render(device) {
  const expectedLength = expand().length;
  const width = 1024, height = Math.ceil((expectedLength + 1) / 128) * 10;
  const imageBytes = width * height * 4, sourceBytes = (expectedLength + 1) * 4;
  if (imageBytes > device.limits.maxStorageBufferBindingSize || imageBytes > device.limits.maxBufferSize) {
    throw new Error("The source image exceeds this adapter's storage-buffer limits.");
  }
  const resources = [];
  const make = (size, usage) => { const b = device.createBuffer({ size, usage }); resources.push(b); return b; };
  device.pushErrorScope("validation");
  let scopeOpen = true;
  try {
    const module = await compileShader(device, shader());
    const main = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const painter = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "paint" } });
    const source = make(sourceBytes, 128 | 4), image = make(imageBytes, 128 | 4);
    const textRead = make(sourceBytes, 8 | 1), imageRead = make(imageBytes, 8 | 1);
    const first = device.createBindGroup({ layout: main.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: source } }] });
    const second = device.createBindGroup({ layout: painter.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: source } }, { binding: 1, resource: { buffer: image } }] });
    const encoder = device.createCommandEncoder();
    let pass = encoder.beginComputePass(); pass.setPipeline(main); pass.setBindGroup(0, first); pass.dispatchWorkgroups(1); pass.end();
    pass = encoder.beginComputePass(); pass.setPipeline(painter); pass.setBindGroup(0, second);
    pass.dispatchWorkgroups(width / 8, Math.ceil(height / 8)); pass.end();
    encoder.copyBufferToBuffer(source, 0, textRead, 0, sourceBytes);
    encoder.copyBufferToBuffer(image, 0, imageRead, 0, imageBytes);
    device.queue.submit([encoder.finish()]);
    const words = await readGPUWords(textRead, sourceBytes);
    const imageWords = await readGPUWords(imageRead, imageBytes);
    const validation = await device.popErrorScope(); scopeOpen = false;
    if (validation) throw new Error(validation.message);
    const rgba = new Uint8ClampedArray(imageBytes);
    for (let i = 0; i < imageWords.length; i++) {
      const v = imageWords[i];
      rgba[4*i] = v & 255; rgba[4*i+1] = (v >>> 8) & 255;
      rgba[4*i+2] = (v >>> 16) & 255; rgba[4*i+3] = v >>> 24;
    }
    const text = decodeWords(words);
    if (decodeImage(rgba, width, height) !== text) throw new Error("Image/source mismatch.");
    return { text, rgba, width, height };
  } finally {
    for (const b of resources) { if (b.mapState === "mapped") b.unmap(); b.destroy(); }
    if (scopeOpen) await device.popErrorScope();
  }
}
function decodeImage(rgba, width, height) {
  if (width !== 1024 || rgba.length !== width * height * 4) throw new Error("Invalid source-image dimensions.");
  const pixel = cell => 4 * ((Math.floor(cell / 128) * 10 + 9) * width + (cell % 128) * 8 + 7);
  let p = pixel(0);
  const n = rgba[p] | (rgba[p+1] << 8) | (rgba[p+2] << 16);
  if (!n || n + 1 > Math.floor(width / 8) * Math.floor(height / 10)) throw new Error("Invalid source-image length.");
  const bytes = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    p = pixel(i+1);
    if (rgba[p+1] !== 165 || rgba[p+2] !== 90 || rgba[p+3] !== 255 || rgba[p] > 127) throw new Error("Damaged source-image byte.");
    bytes[i] = rgba[p];
  }
  return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
}'''
# Put glyphs directly into the generated shader string. They reproduce because
# that string is inside the JavaScript template reconstructed by the GPU.
visual='\n'.join([
    '(() => {','"use strict";', '// The GPU writes this program, paints its bytes, and embeds those bytes losslessly in pixels.',
    'const Q = Object.freeze([~D]);','const CUT = 0;',EXPAND_JS,
    make_shader_js(extra=paint.replace('@FONT@',numbers(font))), HOST, visualrun, PRINT_JS,
    'const api = { id: "visual", data: Q, self: () => expand(), wgsl: shader, decodeImage, run: () => withDevice(render) };',
    'const DEFAULT_GPU = true;',AUTORUN,'})();',''
])
visual_src,visual_q=assemble_quine('04-visual.js',visual)

# The fifth example is generated by a genuine small self-hosting compiler.
exec((ROOT/'compiler/build_compiler.py').read_text(), globals()) if (ROOT/'compiler/build_compiler.py').exists() else None

import subprocess
for p in sorted((ROOT/'quines').glob('*.js')):
    if p.name == '00-minimal.js':
        continue
    shader = subprocess.run(['node', str(p), '--wgsl'], check=True, capture_output=True).stdout
    (p.with_suffix('.wgsl')).write_bytes(shader)

manifest={'schema':'js-wgsl-five-quines/v1','programs':[]}
for path in sorted((ROOT/'quines').glob('*')):
    if path.is_file():
        raw=path.read_bytes()
        manifest['programs'].append({'path':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('\n'.join(f"{r['path']}: {r['bytes']:,} bytes" for r in manifest['programs']))

# The gallery is self-contained. Only the host reads these initial seeds; none
# of the quines introspects the gallery, a script element, or a source file.
programs = {}
for stem in ['01-application','02-polyglot','03-multiquine','04-visual','05-compiler']:
    programs[stem] = {'js': (ROOT/f'quines/{stem}.js').read_text(), 'wgsl': (ROOT/f'quines/{stem}.wgsl').read_text()}
    templates = {'01-application':full, '02-polyglot':poly, '03-multiquine':multi_j+'\n'+multi_w, '04-visual':visual}
    programs[stem]['template'] = templates.get(stem, '\n'.join(listing))
programs['05-compiler']['examples'] = [
    {**entry, 'program':json.loads((ROOT/f"compiler/{entry['name']}.json").read_text())}
    for entry in json.loads((ROOT/'compiler/examples.json').read_text())
]
demo = (ROOT/'lib/demo-template.html').read_text()
demo = demo.replace('__PROGRAMS_JSON__', jsstr(programs) if isinstance(programs,str) else json.dumps(programs,ensure_ascii=True,separators=(',',':')).replace('<','\\u003c'))
demo = demo.replace('__HOST_JSON__', jsstr(HOST).replace('<','\\u003c'))
(ROOT/'demo.html').write_text(demo)
