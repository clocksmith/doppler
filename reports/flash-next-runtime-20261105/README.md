# Flash-Next runtime integration checkpoint

Scope: decoder routing, residual-stream replication, PLE placement, QSA selection, root-mixer logits, coordinated state transactions, bounded table reads and explicit unsupported-mode errors. Integrates conversion commits through 53e79625.

- The 23-file focused run reports 41 passing runner checks (file-level checks and behavioral subtests); the separate profile/text-only run reports 2 passing file checks. See `regression.log` and `profile.log`.
- Apple Metal executes synthetic root-mixer/logits checks through immediate, GPU-output and recorded paths against a closed-form reference within 2e-6. Checks cover last-position selection, rejection of double normalization, ownership and a PLE failure after gating.
- Combined-state rollback, late cancellation, nested chunks and close-during-read use contract doubles for command copies and I/O, not a numerical model.
- The real conversion recipe is lowered through conversion planning, manifest construction, runtime parsing and execution compilation. Every layer's required control projection resolves to f32 in both phases. Text-only metadata is repaired: excluded vision weights no longer leave a vision configuration that makes load parsing fail.
- Strict typing, declaration coverage, unused checks, architecture, browser imports, config single-source and package boundaries pass. The storage range helper enters strict coverage; unchecked inventory shrinks by one and the shard-manager soft ceiling shrinks from 793 to 787 lines.
- `package-inventory.json` audits every shipped delta from a1864d26, including the upstream retained-source converter and recipe. No checkpoint or source-table bytes ship; the minimal Capsule closure remains unchanged.

Limits: the conversion recipe exists, but no converted model artifact or learned-weight Flash-Next execution is established here. Prefix-state exports, embeddings, training, LoRA, padded/multimodal inputs, speculation, GPU-feedback multi-token loops, layer partitions and automatic precision fallback are rejected. GPU checkpoint copies add unmeasured cost; no throughput claim is made. Table mappings and bounded requests are checked; individual row-byte integrity is not established (`rowBytesVerified: false`).

Jupiter acquisition was observed complete: 131 files, 360000192888 bytes, at `/run/media/x/models/qwen38-flash-next-de4b8e4d/source`. Its acquisition receipt reports checkpoint hash/header verification. Source preparation is separate from inference qualification.

Component: `doppler.runtime-source.inference.pipelines.text`, `doppler.runtime-source.loader`, `doppler.runtime-source.storage`, `doppler.runtime-source.config`. Intent: preserved. Boundary effects: declared decoder mechanisms now participate in text execution; range requests explicitly reject whole-file fallback; the conversion recipe declares a text-only artifact.
