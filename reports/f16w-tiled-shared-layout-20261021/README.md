# Register-tiled F16 matmul: conflict-free shared layout (2026-10-21)

`matmul_f16w_f32a_tiled.wgsl` staged both operand tiles row-major in workgroup memory. In the
inner loop the 16 `ty` lanes of a wavefront read B at a stride of 64 floats, so every read hit
one bank (16-way conflict). Tiles are now k-major `vec4` rows: A is broadcast across `ty` and
B is contiguous across `ty`. Global loads and every FMA, in order, are unchanged, so the
output is bit-identical.

## Evidence

AMD Radeon 8060S (RADV STRIX_HALO), Linux 7.0.0-38.

| Check | Result |
|---|---|
| Chrome 146 Vulkan, 11 shapes incl. 1×1×1, 65×63×17, 130×151×2000 ([result](chrome-operator-comparison.json), [script](chrome-operator-comparison.js)) | **Byte-identical** on every shape |
| Same, GPU timestamps, median of 20 | 613×3072×1024: 2.82 → 0.88 ms (3.2×); 613×1024×3072: 2.60 → 0.78 ms (3.3×); 102×3072×1024: 0.66 → 0.17 ms (3.9×); 513×4096×2560: 8.68 → 2.92 ms (3.0×) |
| Node `webgpu` (Dawn/Vulkan), production dispatch geometry, 7 shapes ([output](node-parity.json)) | Byte-identical to the [previous shader](previous-matmul_f16w_f32a_tiled.wgsl); max error 5.3e-7 against an F64 reference |
| End to end: 90 issue-triage requests × 3 `scoreChoices` calls on Qwen3-Reranker-0.6B weights ([summary](continuation-summary.json)) | **90/90 bit-identical** yes/no logits; median 877 → 633 ms per issue |

The end-to-end run used an experiment manifest that also routes the reranker's F16 FFN to this
kernel and prefill attention to `attention_head128_f16kv` (`.cache/`, not committed). Against
the checked-in reranker manifest the same task measured 1,929 ms per issue. That routing is a
separate model-identity change, not part of this commit.

`tests/kernels/matmul-f16w-f32a-tiled-parity.test.js` keeps the F64 tolerance check and accepts
`--reference-shader <path>` for byte-exact comparison with a prior shader.

## Scope

- Ten conversion configurations pin this kernel; their digest is updated (Gemma 3, GLM-OCR,
  LFM2.5, Qwen3-Embedding F16). Checked-in `models/local` manifests keep their existing pins,
  like their other stale pins; qualification fixtures refresh pins explicitly.
- Bit-identical output does not qualify any model or package; it removes a numerical question.
  Speedups are one AMD device; other GPUs need their own timing.

Component: `doppler.runtime-source.gpu.kernels`, `doppler.runtime-source.config`, `doppler.tests`.
Intent: preserved.
Acceptance evidence: this directory, `npm run kernels:check`, `node tests/kernels/matmul-f16w-f32a-tiled-parity.test.js --require-gpu --reference-shader reports/f16w-tiled-shared-layout-20261021/previous-matmul_f16w_f32a_tiled.wgsl`.
Boundary effects: none beyond kernel digest pins in the listed conversion configurations.
