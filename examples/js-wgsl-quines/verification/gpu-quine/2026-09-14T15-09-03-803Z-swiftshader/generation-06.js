const q = "(async () => {\n  if (!globalThis.navigator?.gpu) throw new Error(\"WebGPU is unavailable. Use HTTPS or localhost.\");\n  const adapter = await navigator.gpu.requestAdapter();\n  if (!adapter) throw new Error(\"No WebGPU adapter is available.\");\n  const device = await adapter.requestDevice();\n  let output, readback;\n  try {\n    const code = `\nconst body = array<u32, ${q.length}>(${Array.from(q, c => c.charCodeAt(0) + \"u\").join(\",\")});\n@group(0) @binding(0) var<storage, read_write> result: array<u32>;\nvar<private> cursor: u32 = 1u;\n\nfn put(c: u32) {\n  result[cursor] = c;\n  cursor += 1u;\n}\n\n@compute @workgroup_size(1)\nfn main() {\n  // Write: const q = \"\n  put(99u); put(111u); put(110u); put(115u); put(116u);\n  put(32u); put(113u); put(32u); put(61u); put(32u); put(34u);\n\n  // Quote the body as a JavaScript string literal.\n  for (var i = 0u; i < ${q.length}u; i += 1u) {\n    let c = body[i];\n    if (c == 34u || c == 92u) {\n      put(92u); put(c);\n    } else if (c == 10u) {\n      put(92u); put(110u);\n    } else {\n      put(c);\n    }\n  }\n\n  // Close the string, add a semicolon and a newline.\n  put(34u); put(59u); put(10u);\n\n  // Append the same body as executable JavaScript.\n  for (var i = 0u; i < ${q.length}u; i += 1u) {\n    put(body[i]);\n  }\n  result[0] = cursor - 1u;\n}`;\n    const module = device.createShaderModule({ code });\n    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === \"error\");\n    if (errors.length) throw new Error(errors.map(m => m.message).join(\"\\n\"));\n    const pipeline = await device.createComputePipelineAsync({\n      layout: \"auto\", compute: { module, entryPoint: \"main\" }\n    });\n    const size = (3 * q.length + 16) * 4;\n    output = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });\n    readback = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });\n    const bindings = device.createBindGroup({\n      layout: pipeline.getBindGroupLayout(0),\n      entries: [{ binding: 0, resource: { buffer: output } }]\n    });\n    const encoder = device.createCommandEncoder();\n    const pass = encoder.beginComputePass();\n    pass.setPipeline(pipeline);\n    pass.setBindGroup(0, bindings);\n    pass.dispatchWorkgroups(1);\n    pass.end();\n    encoder.copyBufferToBuffer(output, 0, readback, 0, size);\n    device.queue.submit([encoder.finish()]);\n    await readback.mapAsync(GPUMapMode.READ);\n    const words = new Uint32Array(readback.getMappedRange());\n    const length = words[0];\n    if (!length || length >= words.length) throw new Error(\"Invalid GPU output length.\");\n    const source = new TextDecoder().decode(Uint8Array.from(words.subarray(1, length + 1)));\n    readback.unmap();\n    console.log(source.slice(0, -1));\n    return source;\n  } finally {\n    readback?.destroy();\n    output?.destroy();\n    device.destroy();\n  }\n})().catch(console.error);\n";
(async () => {
  if (!globalThis.navigator?.gpu) throw new Error("WebGPU is unavailable. Use HTTPS or localhost.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter is available.");
  const device = await adapter.requestDevice();
  let output, readback;
  try {
    const code = `
const body = array<u32, ${q.length}>(${Array.from(q, c => c.charCodeAt(0) + "u").join(",")});
@group(0) @binding(0) var<storage, read_write> result: array<u32>;
var<private> cursor: u32 = 1u;

fn put(c: u32) {
  result[cursor] = c;
  cursor += 1u;
}

@compute @workgroup_size(1)
fn main() {
  // Write: const q = "
  put(99u); put(111u); put(110u); put(115u); put(116u);
  put(32u); put(113u); put(32u); put(61u); put(32u); put(34u);

  // Quote the body as a JavaScript string literal.
  for (var i = 0u; i < ${q.length}u; i += 1u) {
    let c = body[i];
    if (c == 34u || c == 92u) {
      put(92u); put(c);
    } else if (c == 10u) {
      put(92u); put(110u);
    } else {
      put(c);
    }
  }

  // Close the string, add a semicolon and a newline.
  put(34u); put(59u); put(10u);

  // Append the same body as executable JavaScript.
  for (var i = 0u; i < ${q.length}u; i += 1u) {
    put(body[i]);
  }
  result[0] = cursor - 1u;
}`;
    const module = device.createShaderModule({ code });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => m.message).join("\n"));
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto", compute: { module, entryPoint: "main" }
    });
    const size = (3 * q.length + 16) * 4;
    output = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    readback = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bindings = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: output } }]
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(readback.getMappedRange());
    const length = words[0];
    if (!length || length >= words.length) throw new Error("Invalid GPU output length.");
    const source = new TextDecoder().decode(Uint8Array.from(words.subarray(1, length + 1)));
    readback.unmap();
    console.log(source.slice(0, -1));
    return source;
  } finally {
    readback?.destroy();
    output?.destroy();
    device.destroy();
  }
})().catch(console.error);
