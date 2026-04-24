# Hardware Assumptions

Referenced by: all inference skills

## GPU Memory Reporting

Do not assume GPU memory limits from WebGPU adapter log lines like `[GPU] amd rdna-3, f16/subgroups, 4.0GB`. On APU systems with unified memory (e.g., AMD Ryzen 395+ with 128GB), the WebGPU adapter reports an artificially low VRAM cap that does not reflect actual available memory. Models that appear "too large" for the reported VRAM may load and run fine.

If inference fails during `loadWeights` with `Device not initialized` or `GPU device not initialized`, do not rewrite that into "likely VRAM pressure" on unified-memory systems. Treat it as a device lifecycle failure first: verify device initialization, device-lost handling, and the first failing buffer allocation or upload before making any capacity claim.

## Timeout vs Memory

When a model times out or is killed during testing:
- Increase the timeout before concluding the model is too large.
- Never skip testing a model based on the adapter's reported VRAM.
- Never make config decisions (kernel variant, dtype) based on untested assumptions about what fits in memory.
