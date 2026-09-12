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
function decodeWords(words) {
  const length = words[0];
  if (!length || length >= words.length) throw new Error("Shader output is empty, truncated, or oversized.");
  const bytes = new Uint8Array(length);
  for (let i = 0; i < length; i++) {
    const code = words[i + 1];
    if (code > 127) throw new Error("Shader returned a non-ASCII source byte.");
    bytes[i] = code;
  }
  return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
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
async function dispatchText(device, code, entryPoint = "main", constants = {}, input = null) {
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
    return { text: decodeWords(words), words };
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
