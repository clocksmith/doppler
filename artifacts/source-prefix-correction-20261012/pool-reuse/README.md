# Reuse before projected allocation pressure

Component: `doppler.runtime-source.memory`. Intent: preserved.

The AMD standalone attachment diagnostic with the accepted-base RoPE extent repair opened successfully but rejected a prefill allocation under the unchanged device ceiling. Its public memory snapshot showed reusable pooled buffers still present. This observation motivated a bounded ownership regression; it did not prove that all model state could fit.

`tests/memory/buffer-pool-device-budget.test.js` demonstrates an independent defect: with 512 bytes already allocated and an exact compatible 256-byte buffer released into the pool, reacquisition first applied the projected *new-allocation* pool policy. It evicted the reusable buffer into deferred destruction and then rejected a fresh allocation against the still-charged device ceiling. `before.log` retains that failure.

The fix checks compatible reuse first and applies allocation-budget trimming only when allocation is actually required. Reuse adds no allocated bytes. No device/pool budget, alignment, precision, model input or GPU completion rule changes. Controls cover soft and hard pool policies, another active owner, pending GPU retirement remaining charged, and successful new allocation only after confirmed settlement. Existing cleanup and device-epoch regressions also pass.

Acceptance evidence: `after.log`, `architecture.log`, and `style.log`. These are CPU fake-device tests, not physical execution or a claim that the attachment prefill now fits. The diagnostic archive and physical replay remain separate. Boundary effects: allocation versus reuse accounting within the existing buffer pool only.
