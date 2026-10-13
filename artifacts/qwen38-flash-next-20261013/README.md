# Flash-Next pinned operator reference preparation

Component: `doppler`. Intent: preserved. Boundary effects: CPU reference
preparation only; no runtime implementation, model acquisition or promotion.

`source-binding.json` binds Qwen/Qwen3.8-Flash-Next metadata at
`de4b8e4d43b917e7706784d8bb445c9af86a3540` and the Transformers qwen4_exp
implementation at `e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`. Every retained
source file has its exact URL, length and SHA256. The binding's `executed:false`
describes initial source acquisition; the subsequent execution receipt is
`reference-fixtures.json`.

Run `python3 artifacts/qwen38-flash-next-20261013/prepare-reference.py` from the
Doppler root with the receipt's CPU PyTorch environment. It validates source
hashes and extracts unchanged upstream definitions through Python AST. Gated
residual fixtures use the actual 2560 hidden width and four residual streams,
with deterministic synthetic weights. Malformed input width rejects. N-gram
fixtures execute upstream address computation for eight tokens, including EOS
boundaries; a separate Python-integer calculation checks all sixteen addresses
per token. A lookup spy replaces the enormous learned embedding allocation;
this tests hashing, not embedding values or learned-model quality.

No checkpoint weight was downloaded. Tokenizer configuration is pinned, but
actual tokenizer vocabulary bytes have not been acquired or qualified. The
metadata declares qwen4_exp, sparse-attention indexer/compression, gated
residuals, n-gram embeddings and 512 experts with ten selected per token.
Existing Doppler operator support establishes none of this compatibility.

Remaining reference work includes sparse attention/indexer, expert execution,
incremental n-gram state, learned-checkpoint comparison, conversion and exact
installed distributed acceptance. Retained upstream LICENSE is part of the
source contract; service/license suitability remains an unresolved release
condition. These fixtures permit parallel source preparation while ordinary
runtime repair continues, without making Flash-Next a dependency for chat.
