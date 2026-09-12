(() => {
"use strict";
const PROGRAM_LENGTH = 55;
const P = Object.freeze([21331,0,52,1,2,20,0,1,3,0,0,1,4,1,0,1,5,1,0,14,2,0,0,3,6,3,4,2,3,4,0,2,4,6,0,4,2,2,5,15,0,0,0,13,3,0,0,1,7,10,0,12,7,0,0]);
function at(a, i) { return i < a.length ? a[i] : 0; }
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
}
function executeBytes(input = P, target = 0) {
  validateProgram(input);
  if (target!==0&&target!==1&&target!==2) throw new Error("Use target 0 (JS), 1 (WGSL), or 2 (Wasm).");
  const r = new Uint32Array(64);
  r[0] = target; r[1] = input.length;
  const output = [];
  const put = c => { if (c>255 || output.length >= 262143) throw new Error("Invalid or oversized compiler output."); output.push(c); };
  const decimal = n => { for (const c of String(n >>> 0)) put(c.charCodeAt(0)); };
r[2]=20;
r[3]=0;
r[4]=1;
r[5]=1;
while(r[2]!==0){
r[6]=(r[3]+r[4])>>>0;
r[3]=r[4];
r[4]=r[6];
r[2]=(r[2]-r[5])>>>0;
}
decimal(r[3]);
r[7]=10;
put(r[7]);

  return Uint8Array.from(output);
}
function execute(input=P,target=0) {
  const bytes=executeBytes(input,target);
  return target===2?bytes:new TextDecoder("utf-8",{fatal:true}).decode(bytes);
}
async function acquireDevice() {
  let gpu = globalThis.navigator?.gpu;
  if (!gpu && typeof process !== "undefined" && process.versions?.node) {
    let binding;
    try { binding = await import("webgpu"); }
    catch { throw new Error("No WebGPU provider. In a browser use HTTPS/localhost. For Node: npm install webgpu"); }
    Object.assign(globalThis, binding.globals);
    gpu = binding.create([]);
  }
  if (!gpu) throw new Error("WebGPU is unavailable. Use a supported browser on HTTPS or localhost.");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter was returned.");
  const info = adapter.info || {};
  const identity = { vendor: info.vendor || "", architecture: info.architecture || "",
    device: info.device || "", description: info.description || "",
    isFallbackAdapter: info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null };
  const device = await adapter.requestDevice();
  return { gpu, device, identity };
}
async function compileShader(device, code) {
  if (typeof code !== "string" || code.length > 2000000) throw new Error("Invalid shader source size.");
  const module = device.createShaderModule({ code });
  const messages = (await module.getCompilationInfo()).messages;
  const errors = messages.filter(message => message.type === "error");
  if (errors.length) throw new Error(errors.map(message => `${message.lineNum}:${message.linePos} ${message.message}`).join("\n"));
  return module;
}
function decodeWords(words, binary = false) {
  const length = words[0];
  if (!length || length >= words.length) throw new Error("Shader output is empty, truncated, or oversized.");
  const bytes = new Uint8Array(length);
  for (let i = 0; i < length; i++) {
    const code = words[i + 1];
    if (code > (binary ? 255 : 127)) throw new Error("Shader returned a non-ASCII source byte.");
    bytes[i] = code;
  }
  return binary ? bytes : new TextDecoder("utf-8", { fatal: true }).decode(bytes);
}
async function readGPUWords(buffer, size) {
  await buffer.mapAsync(1, 0, size);
  try {
    const view = new DataView(buffer.getMappedRange(0, size));
    const words = new Uint32Array(size / 4);
    for (let i = 0; i < words.length; i++) words[i] = view.getUint32(i * 4, true);
    return words;
  } finally { buffer.unmap(); }
}
async function dispatchText(device, code, entryPoint = "main", constants = {}, input = null, binary = false) {
  const size = 1048576;
  let output, staging, inputs;
  device.pushErrorScope("validation");
  let scopeOpen = true;
  try {
    const module = await compileShader(device, code);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint, constants } });
    output = device.createBuffer({ size, usage: 128 | 4 });
    staging = device.createBuffer({ size, usage: 8 | 1 });
    const entries = [{ binding: 0, resource: { buffer: output } }];
    if (input !== null) {
      const values = Uint32Array.from([input.length, ...input]);
      inputs = device.createBuffer({ size: values.byteLength, usage: 128 | 8 });
      device.queue.writeBuffer(inputs, 0, values);
      entries.push({ binding: 1, resource: { buffer: inputs } });
    }
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, staging, 0, size);
    device.queue.submit([encoder.finish()]);
    const words = await readGPUWords(staging, size);
    const validation = await device.popErrorScope(); scopeOpen = false;
    if (validation) throw new Error(validation.message);
    return binary ? { bytes: decodeWords(words, true), words } : { text: decodeWords(words), words };
  } finally {
    if (staging?.mapState === "mapped") staging.unmap();
    staging?.destroy(); output?.destroy(); inputs?.destroy();
    if (scopeOpen) await device.popErrorScope();
  }
}
async function withDevice(work, timeoutMs = 30000) {
  const { gpu, device, identity } = await acquireDevice();
  let timer;
  try {
    const result = await Promise.race([
      work(device),
      device.lost.then(info => { throw new Error(`WebGPU device lost: ${info.message || info.reason}`); }),
      new Promise((_, reject) => { timer = setTimeout(() => { device.destroy(); reject(new Error("WebGPU execution timed out.")); }, timeoutMs); })
    ]);
    return { ...result, adapter: identity };
  } finally { clearTimeout(timer); device.destroy(); void gpu; }
}
/** Memory I/O only. Compilation happens inside the supplied Wasm module. */
async function instantiateSprout(bytes) {
  const { instance } = await WebAssembly.instantiate(bytes);
  const { memory, run, input, output } = instance.exports;
  if (!(memory instanceof WebAssembly.Memory) || typeof run !== "function" ||
      input?.value !== 524288 || output?.value !== 262144) {
    throw new Error("Not a Sprout Wasm module.");
  }
  const compile = (program = null, target = "js") => {
    const targetId = ["js", "wgsl", "wasm"].indexOf(target);
    if (targetId < 0) throw new Error("Use js, wgsl, or wasm.");
    let pointer = 0;
    let length = 0;
    if (program !== null) {
      if (!Number.isSafeInteger(program.length) || program.length > 60000) {
        throw new Error("Invalid Sprout input length.");
      }
      pointer = input.value;
      length = program.length;
      const view = new DataView(memory.buffer);
      for (let i = 0; i < length; i++) {
        const word = program[i];
        if (!Number.isSafeInteger(word) || word < 0 || word > 0xffffffff) {
          throw new Error("Sprout words must be u32 integers.");
        }
        view.setUint32(pointer + i * 4, word, true);
      }
    }
    const size = run(targetId, pointer, length);
    if (size < 0 || size > 262143) throw new Error("Invalid Sprout output length.");
    const result = new Uint8Array(memory.buffer, output.value, size).slice();
    return targetId === 2 ? result : new TextDecoder("utf-8", { fatal: true }).decode(result);
  };
  return Object.freeze({
    compile,
    self: () => compile(null, "wasm"),
    js: () => compile(null, "js"),
    wgsl: () => compile(null, "wgsl"),
  });
}
function printSource(text) {
  if (typeof process !== "undefined" && process.stdout?.write) process.stdout.write(text);
  else if (typeof document !== "undefined") {
    const area = document.createElement("textarea");
    area.readOnly = true; area.value = text;
    area.style.cssText = "width:95vw;height:70vh;font:12px monospace";
    document.body.append(area);
  } else console.log(text);
}
const targetNumber=target=>["js","wgsl","wasm"].indexOf(target);
const api={id:"compiler",program:P,execute,executeBytes,
  self:()=>execute(P,0),wgsl:()=>execute(P,1),wasm:()=>execute(P,2),
  compile:(program,target="js")=>execute(program,targetNumber(target)),
  fromWasm:instantiateSprout,
  run:(target="js",input=null,shader=null)=>{
    const selected=targetNumber(target);
    if(selected<0)throw new Error("Use js, wgsl, or wasm.");
    if(input!==null)validateProgram(input);
    return withDevice(device=>dispatchText(device,shader??execute(P,1),"main",
      {TARGET:selected,EXTERNAL:input===null?0:1},input??[],selected===2));
  }
};
const DEFAULT_GPU=false;
globalThis.Quine = Object.freeze(api);
if (!globalThis.__QUINE_LIBRARY__) {
  const args = typeof process !== "undefined" ? process.argv.slice(2) : [];
  const task=args.includes("--wasm")?Promise.resolve({text:api.wasm()}): args.includes("--wgsl") ? Promise.resolve({ text: api.wgsl() })
    : args.includes("--cpu") || (!DEFAULT_GPU && !args.includes("--gpu"))
      ? Promise.resolve({ text: api.self() }) : api.run();
  globalThis.QuineDone = task.then(result => { printSource(result.text); return result; });
  globalThis.QuineDone.catch(error => {
    if (typeof process !== "undefined") { console.error(error.message); process.exitCode = 1; }
    else { const pre = document.createElement("pre"); pre.textContent = error.message; document.body.append(pre); }
  });
}
})();
