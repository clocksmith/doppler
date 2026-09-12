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
