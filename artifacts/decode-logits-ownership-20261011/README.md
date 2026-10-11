# CPU-logits success ownership

The installed teacher-forced model run retained 196 GPU logits buffers after 200
readback steps. Its caller requested CPU logits, not GPU ownership. The existing
ownership test now reproduces the success-path defect (`before.log.gz`) and verifies
the correction (`after.log.gz`): after successful CPU readback, the decode owner
releases the GPU logits allocation and returns `logitsBuffer: null`.

The same test preserves explicit `_returnGpuLogits: true` ownership transfer,
readback failure cleanup, layer failure cleanup, and borrowed session ping-pong
buffers. This is a synthetic ownership regression; subsequent installed physical
acceptance remains separate.

```sh
node tests/inference/decode-output-ownership.test.js
```

Component: `doppler.runtime-source.inference.pipelines.text`.
Intent: preserved.
Boundary effects: CPU-logits callers no longer receive an unrequested GPU lifetime.
