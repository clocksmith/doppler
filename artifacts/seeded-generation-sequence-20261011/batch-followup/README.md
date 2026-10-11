The first seed fix omitted recorded batch sampling and range-backed PLE stepwise decoding. The retained failing test observes seven zero seeds for seven outputs after three prior outputs. Both owners now derive the internal seed from the committed output index using the existing helper. The regression executes both real loops with only GPU/storage operations stubbed; it compares whole and split batches, retries, first draws, unseeded operation, accepted fractional and large seeds, wrap, immutable options and greedy output. Physical GPU sampling arithmetic remains outside this control.

Validation: `node tests/inference/seeded-batch-generation.test.js`, `node tests/inference/seeded-generation-sequence.test.js`, generator cleanup and hot-vocabulary continuation tests pass. Final combined repository/installed-package checks belong to the concurrent selective-piece snapshot.

Component: `doppler.runtime-source.inference.pipelines.text`. Intent: preserved. Boundary effects: none.
