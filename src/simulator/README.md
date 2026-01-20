# Simulator

Hardware emulation layer for browser-only performance experiments. The simulator
creates virtual GPUs/CPUs, NVLink fabrics, and an emulated VRAM store, then
injects timing delays and memory constraints based on the runtime emulation
config. It is config-gated and only initialized when `runtime.emulation.enabled`
is true.

## What It Simulates

- Virtual GPU and CPU devices with memory budgets and allocation tracking.
- NVLink and NVLink-C2C interconnect timing and bandwidth constraints.
- Emulated VRAM storage backed by OPFS, with RAM fallback.
- Timing model for compute and transfer costs.

## Integration Points

- `src/inference/pipeline/init.js` dynamically imports the simulator and creates
  an emulation context when enabled in runtime config.
- `cli/runners/simulation.js` provides the CLI entrypoint for simulation runs.
- Emulation is optional and is skipped entirely when disabled.

## Key Dependencies

- Runtime emulation schema: `src/config/schema/emulation.schema.js`
- Emulated VRAM store: `src/storage/emulated-vram.js`
- Buffer pool (GPU allocations): `src/memory/buffer-pool.js`
- Debug logging: `src/debug/index.js`

## Limitations

- Not a pure offline simulator: it uses WebGPU and OPFS when available.
- Timing model is approximate; it injects delays but does not emulate kernel
  behavior or scheduling.
- Emulation requires browser APIs (`navigator.storage`, `navigator.gpu`).

## Runtime Config

The simulator is controlled via `runtime.emulation` (see schema and presets).

Example (config snippet):

```json
{
  "runtime": {
    "emulation": {
      "enabled": true,
      "targetChip": "gh200",
      "timingMode": "functional",
      "statsEnabled": true
    }
  }
}
```

## Public API

Main entrypoint is `src/simulator/index.js`. It exports:

- `createEmulationContext(configOverrides)`
- `createEmulationContextForChip(chipType, overrides)`
- `isEmulationSupported()`
- `getEmulationCapabilities()`

## Files

- `index.js` / `index.d.ts`: public API and emulation context.
- `virtual-gpu.js`, `virtual-cpu.js`, `virtual-cluster.js`: device models.
- `timing-model.js`: compute/transfer timing model.
- `nvlink-fabric.js`, `nvlink-c2c.js`: interconnect simulation.
*** End Patch"}"}]}
