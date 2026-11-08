# Flash-Next: bounded host expert reuse on AMD

This is a single-host performance investigation of the existing experimental
Flash-Next F32-KV execution lane. It changes expert acquisition, not weights,
shaders, routing, accumulation, precision or the numerical acceptance policy.
The long-context source-reference gate remains open. These measurements do not
qualify the full model or a browser runtime, publish a model, or promote a recipe.

## Change and result

`RoutedExpertCache` can retain the unchanged assembled Q4K bytes for an individual
expert after its GPU entry is evicted. A subsequent GPU miss uploads that payload
without another storage-reader/materialization round trip. Packed scales, row
padding, expert IDs, working-bank offsets and routing weights are unchanged.

The opt-in `profiles/flash-next-node-host-cache-qualification` extends the existing
F32-KV lane with a **16 GiB host-payload limit** and **repeat-read admission**. The
first source read does not retain a payload. A later source read may admit it.
This avoids collecting one-use prefill payloads. The default host cache remains
disabled. GPU cache, total accounted GPU budget and activation/KV reserve stay at
2, 24 and 4 GiB respectively.

Final paired process order was off → on → on → off (`i1,j1,j2,i2`), three identical
requests per process. Each generated 16 greedy output IDs from the raw five-token
prompt `The capital of France is`. Both arms reset generation between requests
and observe every complete 248,320-value logit row. These are observed costs with
the same observer in both arms, not an uninstrumented throughput claim.

| Request within process | Host cache off: request seconds | Repeat-read cache: request seconds | Cached decode seconds/interval |
| --- | ---: | ---: | ---: |
| First, empty expert caches | 18.60–18.69 | 18.32–18.36 | 0.963–0.970 |
| Second | 18.88–18.97 | 13.68–13.70 | 0.715 |
| Third, retained working set | 18.86–19.01 | 11.07–11.26 | 0.618–0.629 |

Baseline retained decode was 1.025–1.037 seconds/interval. The cached third
request therefore observed about **1.6 tokens/second**, with request time about
41% lower in this fixture. Do not extrapolate to arbitrary prompts or an SLA.
The first prefill was still 3.77–3.91 seconds versus 3.68–3.70 off: a small observed
cost, not a cold-start improvement. OS caches were not cleared or controlled.

The third cached request needed zero expert source-range bytes versus 18.742 GB
off; it still uploaded the same selected bytes. Host payload residency reached
8.193 GiB, below the 16 GiB bound. Accounted peak GPU memory was 13.238 GiB in both
arms. Host RSS, GPU accounting and global GTT are different observations and must
not be added as independent physical allocations on this UMA machine. Staging,
backend/driver caches and metadata are outside the retained-payload limit.

## Correctness and lifecycle

[summarize.py](summarize.py) rehashed the actual retained binary files, checked
their lengths and compared tokens and every logit byte across **44 requests,
704 logit rows and 174,817,280 values**. All were identical. The count includes
the earlier rejected cache variants below; it is not 44 independent workloads.
[comparison.json](comparison.json) preserves every primary timing row.

The existing physical lifecycle harness passed queued cancellation, active-decode
rollback, reset/reuse, paused-consumer cancellation and unload of a paused consumer
with caching enabled. There were no GPU errors or cleanup failures. Unit cases
also cover partial-read failure, cancellation with a live host reservation and
retry, capacity eviction, exact packed-byte recovery and unload during a read.
Evicting a GPU entry still waits for submitted work; host cache hits use
`writeBuffer`'s copied bytes, rather than lending memory to pending GPU work.

## Actual public-API chat smoke

The unchanged public `chatText` harness asked for Paris, 42 and Bonjour twice,
with chat templates and measured token usage. Both arms returned the same answers
and usage, and all six checks passed. On the second cycle:

| Case | Off, seconds | Cached, seconds |
| --- | ---: | ---: |
| Paris, 26 prompt / 2 completion tokens | 10.926 | 7.569 |
| Arithmetic, 28 / 3 | 12.242 | 10.709 |
| Translation, 26 / 2 | 10.787 | 9.039 |

This was one off process followed by one cached process, rather than the paired
raw-prompt design. It is an operational check with descriptive timings, not a
causal cold-start benchmark. The first baseline chat was 26.85 seconds while the
first cached chat was 11.34; uncontrolled OS warmth prevents attributing that
difference to this patch. A larger real conversation can select more experts and
has not been benchmarked here.

## Rejected candidates and wall classification

All candidates and their successful, failed-setup and diagnostic receipts are
retained. No faster-looking subset replaced the baseline.

* **8 GiB GPU cache (`b`)** reduced expert source reads but raised retained request
  time to 25.37–26.01 seconds, versus about 19.97 with 2 GiB (`a1`); it also raised
  accounted peak GPU memory. It was rejected. Source-byte reduction alone was not
  a performance result.
* **Report-only exact-range host wrapper (`d`)** reduced retained requests to
  14.44–14.48 seconds. It demonstrated reuse but copied cached ranges again.
* **Generic Node file cache (`f`)** gave 14.52–14.71-second retained requests, but
  the first request rose to 20.65–20.78 versus 18.54–18.61 off. That broad storage
  change was removed; its patch and tests are archived, not shipped.
* **Expert-owned cache with all admission (`h`)** reached 11.35–11.57-second
  retained requests, but first prefill rose to 5.18–5.22 seconds. Repeat-read
  admission (`j`) keeps the warm benefit with substantially less first-prefill
  overhead. Both admission policies remain explicit configuration.

An isolated timestamp diagnostic of three tokens recorded 744 ms of submitted
compute-pass GPU time within 6,023 ms wall time on the 2 GiB path. The 8 GiB path
recorded 770 ms within 13,263 ms. This classifies a large gap outside compute
passes; it does not label every remaining millisecond as disk wait. Copies,
submission, readback, host scheduling and storage-reader work remain included.
Later runs had zero `/proc/self/io.read_bytes` despite many logical source reads:
OS page-cache hits still pay JS allocation, read and assembly costs. The useful
cache retains the assembled payload at its owner, without extra range copies.

The existing pass observer originally failed in Node because the provider does
not install global encoder constructors. It now discovers prototypes from an
unsubmitted device encoder/pass. It records existing submitted passes without
splitting them; the constructor-free unit case and actual AMD diagnostic passed.
Timestamp instrumentation was absent from the primary runs.

## Pins, receipts and reproduction

Hardware: Radeon 8060S, Node 22.22.1, `webgpu` 0.4.0. The resident 27B provider and
old experimental Flash-Next endpoint stayed loaded during comparisons; the 27B
unit/PID remained active and unchanged. TTM pages limit stayed `16099953`.
Admission receipts record free system/GTT memory before each process; they cannot
prove an idle host for the entire run.

Artifact manifest SHA-256:
`4902a0b730a9310903b00bc0f8f587528fa1af9f37a722d2f9287bafceceec09`.
Baseline source revision was `3f821b78e2a2475bfd6408c9925c0282efc1d48c` plus the
explicit experimental patches pinned by each source archive and per-file receipt.
Final repeat-admission archive SHA-256:
`1fe6f34cf394852c620c645c5190c0dafc6e921db410e700b4c4a164f529582e`.
The candidate-server source archive, based on main through `d1963e38`, has SHA-256
`d9e789080324d5437209ad6b211aef9c1583e27a0e41523622ab2fb90c337155`.

Raw logit files, source archives, original JSON/logs and a hash inventory are at
`/run/media/x/models/qwen38-flash-next-de4b8e4d/qualify/cache-budget-20261107`.
The 832 retained files total 729,202,019 bytes, without model weights. The checked
compressed [inventory](raw-inventory.json.gz) and `receipts/` copies retain the
small observations in Git. `retain.py` refuses to overwrite differing evidence.
`gpu-budget` contains A–D; `generic-host` E–F; `expert-all` G–H; `expert-repeat` I–J,
chat and lifecycle receipts. Run:

```sh
python3 summarize.py /path/to/cache-budget-20261107 comparison.json
```

For execution, extract the matching source archive in an isolated directory,
provision the same Node provider, artifact and explicit run JSON, then run:

```sh
DOPPLER_ROUTED_EXPERT_COST=/absolute/run.json node tests/integration/routed-expert-cost-physical.test.js --require-model
```

Each run JSON pins inference/configuration, observer files, model, output path
and sample count. Provision enough memory and run the recorded admission check
before a physical run. The C/D wrapper reproduction also replaces that test file
with [host-prototype-harness.js](host-prototype-harness.js), with its separately
retained wrapper; it is a rejected prototype, not production code.

## Experimental endpoint

The candidate HTTP endpoint passed identity, short chat, streaming, follow-up,
concurrent refusal, client cancellation and reuse, and the output cap; see
[candidate-http-probe.json](candidate-http-probe.json). Only after those checks,
the two owned experimental processes were gracefully retired and the cached
build was opened at `127.0.0.1:8989`. The 27B unit/PID and TTM setting were checked
unchanged. [deployed-route-probe.json](deployed-route-probe.json) records the new
route answering Paris, with 26 measured prompt and 2 completion tokens; the
first request took 11.62 seconds. This is a local process, not a public
deployment, npm release or Reploid provider. The compressed transition receipt
pins its source archive, configuration and owned process IDs.

## Development checks

Four focused test files passed: routed cache ownership/bytes, resolved profiles,
F32-KV policy isolation, and observer ownership. Source typing, unused/style,
architecture, config single-source, browser imports, unchanged capsule runtime
closure and installed-package smoke passed. These checks do not replace model
qualification. Logs are retained alongside this report.

Component: `doppler.runtime-source.loader`, `doppler.runtime-source.config`,
`doppler.tests`, `doppler.internal-tools`, `doppler.docs`.
Intent: preserved. Boundary effects: explicit optional host cache policy and
statistics; no storage backend or inference-selection authority change.
