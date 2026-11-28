# Full-parameter GPT training in a browser

This development benchmark compares Doppler’s dense trainer with the upstream TabGPT WebGPU trainer. Both start from the same 23,994,240 FP32 parameters on the same physical GPU. Both train on the same 32 batches of 4 × 128 next-token targets. Validation uses four fixed batches from a separate Bokmål validation shard. The benchmark pins the model and data sources to [TabGPT revision 7e03875](https://github.com/jbingen/tabgpt/tree/7e0387500f0f64bddf3e68c341508f3f4454cc81). [Workload configuration](workload.json)

## Apple M3 results

Chrome 158 ran three paired FP32 trials. Across all 96 matched training steps, the largest absolute difference in cross-entropy was below 0.0000005. The two independent PyTorch fixtures and the full-model comparison of all 62 parameter tensors passed. [FP32 measurements and per-tensor checks](../../reports/training/tabgpt-24m-apple-m3/fp32.json)

| Trainer | Training tokens/s | Peak live WebGPU buffers | Held-out loss, start → step 32 |
| --- | ---: | ---: | ---: |
| Doppler FP32 (three-trial median) | 3,827 | 601 MiB | 9.87537 → 8.93315 |
| TabGPT FP32 (three-trial median) | 3,400 | 875 MiB | 9.87537 → 8.93315 |
| TabGPT mixed precision (one run) | 5,432 | 575 MiB | 9.87535 → 8.93366 |

The throughput measurement covers 30 timed updates per run. Each trajectory processes 16,384 training targets including warmup. Validation covers the same 2,048 held-out targets before and after training. Upstream mixed precision was the fastest configuration measured on this device. [Mixed-precision measurement](../../reports/training/tabgpt-24m-apple-m3/stock-mixed.json)

## Run

Use a Doppler checkout with its Node dependencies installed, Node 22.13+ with TypeScript stripping, and Chrome with physical WebGPU. Run the following from the checkout root. The scratch directory holds the upstream program, one training shard, one validation shard, and local receipts. No pretrained weights are required.

```sh
scratch=$(mktemp -d)
git clone --filter=blob:none --no-checkout https://github.com/jbingen/tabgpt.git "$scratch/upstream"
node benchmarks/tabgpt/prepare-reference.js "$scratch/upstream" "$scratch/assets"
node benchmarks/tabgpt/run.js "$scratch/assets" "$scratch/full.json" all
node benchmarks/tabgpt/run.js "$scratch/assets" "$scratch/stock-mixed.json" stock-mixed
```

The optional last command runs upstream mixed precision on a device with `shader-f16`. The main comparison uses FP32 throughout. The upstream browser app also adapts its batch size to the machine; this comparison pins batch size and reads loss every step for both implementations.

## How to read the receipt

- **Independent check.** Two tiny two-block models are compared with PyTorch CPU autograd, including gradient clipping, Adam moments across consecutive updates, tied embeddings, and uneven dimensions. The fixtures include reference versions, sources, sampled coordinates and tolerances.
- **Full-model check.** Compare every initial element and each of the first two updates in all 62 named parameter tensors against upstream. Each tensor’s RMS update error must be within the absolute and relative tolerances in the workload. Record the largest individual discrepancy separately. Zero-initialized residual projections make the second update important. Hardware floating-point reductions can differ in their least significant bits.
- **Training and validation.** Each trainer starts a fresh session in each paired trial. Alternate order over three trials. Exclude two warmup steps, time 30 steps, count one token per target, and wait for each optimizer update and loss readback. The timer includes batch upload and excludes token-window sampling, model construction, validation, and parity readbacks. Check every training loss, and the starting and ending validation losses, across each matched pair.
- **Memory.** Count peak live bytes requested through WebGPU buffer allocations during each single-model run. Browser process memory and driver-internal allocations are outside this counter. Initialization is timed separately after loading the reference assets.
