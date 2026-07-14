# Gated SiLU Runtime Correction

Doppler's split gated SiLU path was incorrect. It computed
`SiLU(gate) * SiLU(up)` instead of the SwiGLU operation
`SiLU(gate) * up`. The first confirmed affected model is the Qwen 3.5 9B
F16 artifact used for V12 adapter portability.

Commit `358653e7` fixes dense and MoE split callers and makes the wrapper fail
closed when a gated caller omits the input-branch activation contract. The
committed-code V12 rerun now passes the unchanged gate for the base and seeds
11, 29, and 47, with exact completions and logit cosines above `0.9997`.

The latest npm package is still `doppler-gpu@0.4.8`. Its published tarball was
inspected directly and contains the faulty default and caller. A `0.4.9` patch
release and downstream browser-bundle redeploy are required. Model weights,
RDRR artifacts, tokenizers, and LoRA adapters do not require republishing.

The confirmed scope is the Qwen 3.5 9B F16 split dense path. Split SiLU MoE
callers shared the defect and are corrected in source. Fused FFN, GeGLU,
ungated SiLU, and current Q4K quickstart claims require path-specific
qualification before being labeled affected or unaffected.

The full 433-file unit suite, clean release gate, export parity, public package
boundary, and packed-package smoke pass. Publishing and deployment have not
occurred. The machine-readable receipt is
`gated-silu-runtime-correction-2026-07-13.json`.
