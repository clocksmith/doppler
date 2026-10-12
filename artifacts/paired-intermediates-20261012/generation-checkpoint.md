# Generation integration checkpoint

Component: `doppler.runtime-source.inference.pipelines.text`
Intent: preserved; implements the explicitly approved paired-intermediate exception.
Boundary effects: generator storage, kernel dispatch, observation and cleanup. Reploid's accepted package is unchanged.

This source checkpoint does not qualify an installed archive or complete model. Full 55-prefix comparison, both physical placements, signed opening, concurrent long requests and memory acceptance remain open. Document-answer quality is independently unresolved.

The generator now propagates the bound activation representation into embedding, prefill, ordinary/recorded decode, observations and recovery buffer allocation. Decode buffers distinguish paired storage from scalar F32 and report padded physical allocation. Unsupported paired prefix overrides and embedding extraction reject explicitly. Public logits remain scalar F32. Resident partition carries already use the validated activation byte contract and required no new transport mechanism.

Prefill cleanup now covers failures in base/override probes, preparation and debug readback; cleanup errors preserve the original cause. Follow-up auditing of the older decodeStep sampling path remains in progress.

Fresh checkpoint checks passed:

- `npm run typecheck:source`
- `npm run source:architecture:check`
- `npm run source:style:check`
- `node tests/inference/generator-paired-storage.test.js` (26 paired/scalar orchestration cases)
- `node tests/inference/decode-buffer-representation.test.js`
- `node tests/inference/generator-activation-representation.test.js`
- `node tests/inference/paired-execution-plan.test.js`

Additional focused checks passed during implementation: seeded batch generation, decode output/sampling ownership, prefill output ownership, resident partition lifecycle and execution-plan tests. These use contract fixtures; they do not replace physical inference.

Physical recurrent controls passed on Mac and AMD; full-attention controls passed on Mac. Corresponding operator receipts retain their scopes and failed predecessors. AMD full-attention replay is pending at this checkpoint. The complete paired graph is admitted by a bounded two-layer fixture; whole-model execution has not yet passed.
