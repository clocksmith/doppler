# Allocate KV only for attention consumers in a hybrid model

Component: `doppler.runtime-source.inference.pipelines.text` and KV ownership. Intent: preserved.

The exact AMD attachment replay of diagnostic archive `7ede3796` retained 48 KV allocations across all 24 layers (402,653,184 bytes), although the declared model has six full-attention and eighteen linear-attention layers. Both execution owners prove the missing consumer boundary: `layer.js` routes linear layers to `runLinearAttentionLayer` with recurrent runtime/current position and no KV cache; `ops.js` does likewise for the execution-plan path before conventional attention. `createKVCache` nevertheless allocated the full layer count.

The factory now excludes declared linear-attention layers from the allocation count and exposes their remaining original model indices through the existing scoped-cache proxy. Only allocation/index mapping changes. Precision, geometry, context, model inputs, budgets and recurrent state are unchanged. Mixed geometry/shared-KV allocation keeps its existing owner and behavior. Incomplete hybrid layer declarations fail before allocation.

Tests failed before the fix with four allocated layers where two have KV consumers. CPU and fake-GPU controls now cover compact full/partition allocation, original-index get/update/GPU bindings, position advance, clone, reset isolation, no KV storage for an all-linear partition, immutable index metadata, and rejection of excluded/out-of-range access. Browser KV-byte evidence uses original model indices rather than relabeling compact storage. The opt-in physical partition test now expects the declared KV consumers; it was not run as part of these CPU checks.

Acceptance evidence: six focused test files in `after.log`, with `before.log`, declaration/type checking, source style and architecture logs. These tests establish ownership/index behavior, not physical numerical qualification. The expected reduction for the observed model is 301,989,888 allocated bytes; a separately identified public-package replay must measure it and actual generation.

Boundary effects: existing KV views and original-index diagnostic labels. No package promotion or deployment.
