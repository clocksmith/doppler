# Protected backend plan (Phase A)

Status: research. Nothing here enables `requester-only` execution. `assertExecutionBackendAvailable()` still
rejects every cryptographic descriptor, and Reploid still answers `CONFIDENTIAL_EXECUTION_UNAVAILABLE`. No
cryptographer has reviewed the design, the parameters or the code.

Goal: a protected backend in which the machine doing the work cannot read the input, for Reploid's
`confidentiality: "requester-only"`. Reploid's contract for that mode includes collusion of **all** remote parties.

## 1. Survey

### What Doppler already has

| Piece | What exists | What it proves | Why it is not the backend |
| --- | --- | --- | --- |
| `src/inference/execution-backend-contract.js` | Versioned descriptor (`plaintext` / `attested` / `cryptographic`, opaque input and output for protected kinds), exact model support check, state namespace without prompt digests | A protected descriptor validates; availability is refused for anything but plaintext | Scaffolding only. A descriptor for this plan's prototype validates and is refused (checked: `validate ok`, then "Attested and cryptographic backends are unavailable") |
| `tools/privacy-demo/` | BFV (node-seal 7.0.0 / SEAL) eight-symbol bigram, separate client and evaluator processes, four autoregressive steps, full client ciphertext replay before decryption | A complete client-in-the-loop feedback loop with no plaintext feedback; a demonstrated selective-failure attack on reference-audit modes, closed by replay | Not a transformer; the client replays the whole circuit (no outsourcing) |
| `tools/privacy-concrete/` homomorphic route | Concrete ML GLWE linear layers for tiny-gpt2 (2-dim test checkpoint) and all 169 linear modules of Qwen2.5-0.5B over four steps, separate processes, client replay or sampled checks before decryption | Encrypted linear algebra on real checkpoints with exact-ciphertext replay | **Hybrid**: the client owns embeddings, attention, norms, activations, sampling and the weights it replays with; 134 MB evaluation key per run; stopped by the owner on 2026-11-17 |
| `tools/privacy-concrete/SHARES.md` | Two-server additive shares of each activation row; providers multiply by public weights; probe-row integrity check | Exact, light requester for linear layers, no hardness assumption | Privacy fails completely if the two providers pool views, so it cannot meet `requester-only`'s all-remote collusion; the requester still runs the non-linear steps and stores the embedding table |

None of these keeps every operation off the requester while tolerating collusion of all remote parties.

### Perseus (the owner's reference)

- Paper: Zirilli, Marincione, Kornaropoulos, Ateniese, Rodolà, *Perseus: A Full-Stack System for Homomorphic Model
  Inference*, IACR ePrint 2026/2735 (arXiv 2611.04488). Code: `github.com/gladia-research-group/perseus`, read at
  commit `d1728ff6c2011533f72c81d063da4c141d2e262f`. **Licence: Business Source License 1.1**, non-commercial use only
  until 2030-08-18 (then GPL-2.0). It was read, not run: it needs an NVIDIA sm_120 (Blackwell) GPU and CUDA 13.2.
- Scheme: RNS-CKKS, ring 2^16, a 32-bit RNS chain (51 + 13 primes, log QP 1734, dnum 4), complex-slot "Cachemir"
  packing, half of each key-switching key regenerated from a ChaCha12 seed on the GPU, a packing- and
  magnitude-aware bootstrap placer, optional CKKS-to-TFHE lookups for sign/argmax. Bootstrapping uses a sparse
  encapsulation secret of Hamming weight 32. The README claims every key estimated at or above 128 bits.
- Numbers (README, one RTX PRO 6000, 96 GB): GPT-2 small decode 1.55 s/token at a 512-token context, 416 bootstraps
  per token, 63.3 GiB peak GPU memory; Llama-3-8B decode 6.11 s/token (53.2 GiB). Times exclude key generation,
  encryption and decryption. The paper reports GPT-2 perplexity 26.46 encrypted vs 26.47 plaintext.
- Client/worker split (`python/perseus/serve/gpt2_decode.py`, `client/__init__.py`): the client generates every key
  the server's plan requests and sends one key bundle; per token it encrypts block 0's input row, and decrypts the
  returned logits (and, with `--argmax`, CutMax's one-hot). The server opens its session from the bundle and asserts
  `has_secret` is false. The server loop has no request channel back to the client;
  `python/tests/test_served_no_oracle_cpu.py` checks that a server-sent `ask` (decrypt/fit request) ends the session
  with zero bytes returned.
- Gaps the owner named, confirmed in the code: the served client reads its input rows from the reference oracle
  (`oracle_io.read_teacher_forced_inputs`), so it is teacher-forced and the client holds embedding rows; nothing
  checks that the worker computed the right answer. Autonomous encrypted generation exists only in-process
  (`examples/gpt2/run_generate.py`: encrypted CutMax plus encrypted feedback embedding), where the same process
  holds the secret key. The paper states no threat model and has no integrity mechanism.

### Alternatives

| Family | Examples (primary numbers) | Fits `requester-only`? |
| --- | --- | --- |
| Single-key CKKS, non-interactive | NEXUS (NDSS 2025): BERT-base 857 s on 32 CPU threads, 37 s on four A100s, 164 MB, server-side argmax. EncryptedLLM (ICML 2025): GPT-2 small ~68 s/token on an A100, CPU on the order of 10^4 s. Perseus above | **Yes**: one evaluator, any collusion, the client only encrypts and decrypts |
| CKKS with client-aided non-linear layers | THE-X (ACL Findings 2022): client decrypts, applies ReLU/max, re-encrypts every non-linear layer; BERT-tiny | Confidential, but the client answers evaluator-chosen decryptions every layer: the pattern the owner rules out |
| Two-party HE + MPC | Iron (NeurIPS 2022), BOLT (S&P 2024: BERT-base 91-185 s LAN, 25.7-59.6 GB), BumbleBee (NDSS 2025: BERT-base 2.55 min, 6.4 GB; GPT-2 one token 1.48 min, 2.77 GB) | The client is a full computing party with GB-scale traffic; built to also hide the server's weights, which Reploid does not need |
| Multi-server secret sharing | PUMA (3 parties, LLaMA-7B 200 s/token), SIGMA (2 servers + dealer, GPT-2 1.51 s, Llama2-13B 37.6 s), Gupta-Katz-Miers ePrint 2026/1849 (two non-colluding servers, open weights, 179 MB client storage for Llama 2-70B), Doppler's SHARES route | **No**: all assume non-colluding servers |
| TEEs | H100 confidential computing (<7% overhead, HBM not encrypted, trusts NVIDIA roots), Apple Private Cloud Compute (Apple silicon, transparency log) | Excluded from the cryptographic tier by the owner; trust rests on the vendor and side-channel resistance |
| Verifiable FHE | Viand et al. 2023 (key recovery by a malicious server without integrity); Thibault-Walter CCS 2025: a proof of one TFHE bootstrap takes ~20 min | Integrity only; far too slow today |

Libraries: Lattigo v6.2.0 (Apache-2.0, pure Go, CKKS bootstrapping, compiles to WebAssembly), OpenFHE (BSD-2,
bootstrapping and scheme switching; WASM build is a Node-only work in progress), SEAL / node-seal (MIT, no
bootstrapping), TFHE-rs and Concrete (BSD-3-Clause-Clear, but Zama's patents are licensed only for non-commercial
use; the TFHE-rs JS API does keygen/encrypt/decrypt only), Perseus (BSL, non-commercial, CUDA Blackwell only).

## 2. Design

The smallest design that honestly meets `requester-only` for a small model generating several tokens:

- **Scheme**: single-key RNS-CKKS. One evaluator (or two, for integrity) holds public weights and public evaluation
  keys; the requester holds the only secret key. Confidentiality rests on RLWE and holds when every remote party
  colludes. Library: **Lattigo v6.2.0** for both parties (Apache-2.0). Its requester operations (key generation,
  encryption, decryption, serialization) build for `GOOS=js GOARCH=wasm` and ran under Node's WebAssembly runtime
  on a small test ring (11.5 MB module); browser speed and memory at N = 2^16 are not measured. Perseus is the performance reference, not a dependency, because of its licence and GPU.
- **Model**: a research fixture, not a catalogued Doppler model: one pre-LN GPT-2 block (d=16, one head, MLP 64,
  GELU tanh, tied head), 32-symbol alphabet, 8-position context, trained here on letter progressions
  (`tools/privacy-protected/train_tiny.py`, identity
  `sha256:90d4efb8e2e05df08082a2dc1a2d8ad49479e1750467f70ee2b0759d3578ae92`). GPT-2 small is not feasible at usable
  latency on fleet hardware (section 4).
- **Requester computes**: tokenization (the alphabet), key generation, a one-hot encryption per input token,
  decryption of one 32-value logits ciphertext per step, argmax. It holds **no weights** and loads no model file.
- **Evaluator computes everything else on ciphertext**: embedding lookup (encrypted one-hot times public
  per-position tables of the token embedding, first layer norm, Q, K and V with the output projection folded in),
  attention scores against its encrypted KV cache, softmax (Chebyshev exp and inverse), residual, both layer norms
  (Chebyshev inverse square root), MLP with Chebyshev GELU, tied LM head. Weights, biases and norm gains are folded
  into public plaintext matrices.
- **Round trips**: one per generated token. The first message carries every prompt token; each later message
  carries the previous token. The next token is chosen on the requester and sent freshly encrypted; nothing derived
  from a decryption is sent in the clear. Moving argmax and feedback onto the evaluator (Perseus's CutMax plus
  encrypted feedback) would reduce this to one round trip per request at the cost of more depth; it is a later step.
- **Admission before upload**: the evaluator's manifest must state exactly the pinned profile digest, model
  identity, rotation set, input level and slot layout before any key or ciphertext is sent.
- **Integrity (research protocol)**: two independently operated evaluators run the same deterministic circuit on
  the same ciphertexts and keys; the requester requires byte-identical results (SHA-256) before it deserializes or
  decrypts anything. A disagreement closes the route with no decryption. Rejection depends only on what the
  evaluators returned, never on decrypted values, so it is not a selective-failure oracle. This detects one
  dishonest evaluator; two that collude can return the same wrong answer (confidentiality still holds). It does not
  detect approximation error: both evaluators compute the same pinned approximate circuit. Its cost is a second
  evaluator and a second upload; the requester's own cost is hashing. Client replay (as in `privacy-demo`) would
  need the weights, which this design rules out.
- **What leaks** to the evaluators: model and profile identity, the evaluation keys, the prompt length (number of
  ciphertexts in the first message), the fixed number of generated tokens (public schedule, no early stop),
  ciphertext sizes (fixed), timing per step (depends on position; not shown to be independent of data, and Lattigo
  is not constant-time), and network metadata. Not leaked: tokens, logits, activations. The requester never returns
  a decrypted value, so CKKS's IND-CPA-D key-recovery attack (Li-Micciancio) needs the user to publish outputs and
  then only learns argmax decisions; fresh keys per session limit it further. Inputs outside the calibrated
  intervals make the approximations diverge silently (section 3).

## 3. Prototype (`tools/privacy-protected/`)

`requester`, `evaluator`, `audit`, `compare` and `bootstrap-probe` are subcommands of one Go binary; the requester and each evaluator
run as separate processes over TCP. `evaluator.go` never constructs a secret key, encryptor, decryptor or key
generator, and `requester.go` never loads a model; `TestSourceBoundaries` checks both, and that the requester only
ever waits for `manifest`, `ready` and `result`. Profile `research-n16`: N = 2^16, 64 slots, Q = 50 + 36 x 32
bits, P = 7 x 61 bits (log QP 1629), dense ternary secret, rotation keys for 1, 2, 4, 8, 16, 32 only.
`dev-insecure-n12` is the same chain on N = 2^12, insecure, for fast tests; the requester refuses it without
`--allow-insecure-dev`.

```sh
cd tools/privacy-protected
python3 train_tiny.py --out tiny-model.json         # optional: regenerates the committed fixture
go test -count=1 .                                    # source boundaries, insecure-profile refusal
DOPPLER_PROTECTED_E2E=1 go test -count=1 -v .         # + process tests on the insecure ring (about 1 minute)
go build -o /tmp/pp . && /tmp/pp audit                # plaintext approximation audit
/tmp/pp evaluator --ready-file /tmp/a &  /tmp/pp evaluator --ready-file /tmp/b &
/tmp/pp requester --evaluator $(cat /tmp/a) --evaluator $(cat /tmp/b) \
  --model-identity sha256:$(shasum -a 256 tiny-model.json | cut -d' ' -f1) --prompt hkn --generate 4 --report r.json
/tmp/pp compare --report r.json                       # after the route closed: exact and approximate references
```

### Measured

One run on 2026-11-24 (23:03:21 to 23:09:44 UTC), Apple M3 with 8 cores and 24 GB, the requester and both
evaluators on that one host over loopback TCP, profile `research-n16`, fresh prompt `hkn`, four generated tokens.
Evidence: [`tools/privacy-protected/evidence/research-n16-hkn.json`](../tools/privacy-protected/evidence/research-n16-hkn.json).
One functional run, not a latency distribution; the two evaluators shared the host's cores and memory bandwidth.

| Measure | Value |
| --- | --- |
| Tokens | `qtwz` (16, 19, 22, 25): equal to the approximate (pinned) reference **and** the float64 reference |
| Logit error | at most 0.257 against the approximate circuit, 0.794 against float64; smallest encrypted top-1 margin 8.58 |
| Requester weights | none: it loaded the alphabet, the profile and the model identity only |
| Requester decryptions | 4, each after both evaluators' results were byte-identical |
| Requester key generation | 6.4 s for 1 relinearization + 6 rotation keys, generated, sent and dropped one at a time |
| Evaluation keys | 1,937,800,240 bytes per evaluator per session (3.0 s on loopback to both) |
| Requester peak RSS | 793 MB |
| Upload per token | 38,798,206 bytes per evaluator (one ciphertext at the top of a 37-prime chain); prefill 116 MB for 3 tokens |
| Download per token | 2,097,486 bytes per evaluator |
| Requester encrypt / decrypt / verify | 0.10-0.42 s / 2-10 ms / 2-3 ms per step |
| Round trip per step | 127 s (prefill), then 80, 59 and 106 s; evaluator compute is almost all of it |
| Whole request | 383 s for 4 tokens |
| Evaluator | 5.9-6.0 GB peak RSS; 260-276 s CPU each; over the 4 steps 390 rotations, 70 ciphertext multiplications, 1,024 plaintext multiplications and 34 polynomial evaluations; output at level 1 of 36 |
| Dual-evaluation cost, reported separately | a second evaluator (260 s CPU, 5.9 GB), a second key upload (1.94 GB) and a second copy of every ciphertext (0.23 GB); requester hashing 10 ms in total |

Process tests on the insecure development ring (`DOPPLER_PROTECTED_E2E=1`, all pass): two honest evaluators agree
byte for byte; an evaluator that sends a `decrypt-request` ends the session with zero decryptions; one evaluator of
two that flips a bit closes the route with zero decryptions and no output; a result bound to the wrong step is
refused with zero decryptions.

### Approximation audit (plaintext, `audit`)

Intervals are fixed and public (exp on scores [-8, 9] degree 15; inverse on [0.08, 4] degree 15; inverse square root
for the second norm degree 15 and for the final norm degree 31; GELU on [-13, 8] degree 31). Training kept each
attention row's largest score in [6, 8] so the softmax denominator stays inside the inverse's interval; this is an
encryption-friendly training constraint, like Perseus's HEAT checkpoints. Over all 78 task prompts and four greedy
steps the approximated circuit picks the same tokens as the float64 model (largest logit error 1.02 against a
smallest top-1 margin of 7.35). Over all 17,576 three-letter prompts, 443 (2.5%) generate a different token
sequence, 132 push some input outside an interval, and 512 logit values become non-finite. Out-of-distribution prompts
are therefore a silent-accuracy risk, not a confidentiality one.

## 4. Feasibility verdict

Measured on the same Mac with `pp bootstrap-probe` (Lattigo's example bootstrapping parameters, N = 2^16,
2,048 slots, 10 residual levels): one bootstrap took 11.0 s in a first scratch run (same parameters, not retained) and 22.2 s
in the retained run under memory pressure; bootstrapping keys were 7.9 GB serialized; peak RSS 6.7-10.5 GB
([evidence](../tools/privacy-protected/evidence/bootstrap-probe-n16.json)).

- **Tiny model, leveled, CPU: works, not usable for chat.** One GPT-2-shaped block at d = 16 fits in 35 levels
  without bootstrapping and generates autonomously at about 1-2 minutes per token on a laptop CPU, with ~2 GB of
  evaluation keys per evaluator and ~39 MB uploaded per token.
- **GPT-2 small at usable latency on the fleet: no.** Perseus needs 416 bootstraps per token and 63.3 GiB of GPU
  memory on a Blackwell GPU to reach 1.55 s/token. The fleet has no such GPU (Perseus targets CUDA sm_120 only;
  the AMD GPU host is unsupported and off limits; the Jetson AGX device is sm_87 with 32-64 GB shared). On CPU, 416 bootstraps at
  11-22 s is 1.3-2.5 hours per token for bootstrapping alone, consistent with EncryptedLLM's ~10^4 s/token CPU
  baseline. Bootstrapping keys add ~8 GB per requester session, which breaks "light requester" on bandwidth unless
  keys are reused across sessions and seed-compressed (Perseus halves them that way).
- **Realistic first target**: earn the capability bit on this tiny fixture across two machines (section 6): it
  satisfies every listed condition, at research scale. Next, a 2-4 layer d = 64 encryption-friendly model with
  bootstrapping on a large-memory CPU evaluator, at minutes per token. Seconds-per-token on a real catalogued model
  needs a Blackwell-class GPU evaluator and either a Perseus commercial licence or an own GPU CKKS implementation.
- **Cheap wins before then**: seeded ciphertexts and keys (halve upload and keys), encrypting the one-hot at the
  lowest usable level, caching encoded plaintext diagonals, packing all attention scores into one ciphertext (one
  exp evaluation instead of one per position), and moving argmax and feedback onto the evaluator.

## 5. Reploid protocol sketch

- A new module beside `packages/reploid/src/mesh/peer-generation.js`, for example `mesh/protected-generation.js`,
  with its own message family (`reploid:protected/v0:*`) on the swarm transport and its own protocol version. It
  shares no handler, journal or log with plaintext generation.
- `server/fleet/browser-bridge.js` keeps refusing `requester-only`: its JSON body is plaintext to that host.
  Protected requests start in the process that owns the keys (the browser tab or the native requester itself).
- **prepare**: the route names backend `doppler-protected-ckks/v0`, its protocol identity, one or two evaluator
  peers, and disclosures `contentReadableBy: { requesterDevice: true, participants: [] }` with metadata `model-id`,
  `timing`, `length` (prompt length and generated count). No credit hold, nothing sent.
- **Identity, then admission, then upload**: the evaluator's signed advertisement (existing mesh peer identity)
  names the backend, profile digest, model identity, rotation set and capacity. The requester checks the signature
  and that every field equals its pinned profile, then obtains a signed admission bound to session id, profile
  digest, model identity, schedule (prompt length, generated count) and key-epoch id. Only then are evaluation keys
  and ciphertexts sent. For dual-evaluation integrity the two evaluators must be distinct enrolled owners; the
  receipt records that operator separation is asserted, not proven.
- **Run**: keys once per key epoch, then the prefill and decode steps; results compared before decryption. The
  requester answers no evaluator request; any other message kind ends the session.
- **Separate state, logs and recovery**: protected sessions live in their own journal holding session id, sequence
  numbers, key-epoch id and result digests, never ciphertexts after completion, tokens or text. The secret key lives
  only in the requester's own key store (encrypted at rest), never in config snapshots, receipts or bus messages.
  Evaluator logs carry sizes, timings and digests. Recovery restarts a fresh session on the same or a new key
  epoch; it never retries as executor-readable or local, and never resumes a half-finished decode on another
  evaluator.
- **Receipt**: `evidence.confidentiality.verified` gains `{ claim: "cryptographic-protection", by: "requester",
  method: "requester-held-key/<protocol>" }` only once the review below is done; `integrity.verified` records
  `dual-evaluation-digest` when used. Until then the capability stays unadvertised.

## 6. Two-machine proof plan

1. Pin a commit of this branch; build the binary for each machine. Evaluators get the public fixture; the
   requester gets only the profile, the model identity and the alphabet.
2. Placement, CPU only and inside agreed windows: requester on the Mac; evaluator A and evaluator B on two other
   fleet CPUs (not the AMD GPU host). Check first that both evaluators produce byte-identical results across their CPU
   architectures (Go may fuse multiply-adds differently on arm64 and amd64, which would change plaintext encodings);
   if they differ, pin both to one architecture or precompute the encoded diagonals as integers with published
   digests.
3. Transport: wrap the TCP stream in an authenticated encrypted channel bound to the enrolled evaluator keys (reuse
   `privacy-concrete/shares_network.js`'s Ed25519 + X25519 + ChaCha20-Poly1305 transcript, or pinned SSH).
4. The owner chooses the prompt at run time. Four generated tokens; `compare` runs after the route closes and must
   show encrypted tokens equal to the approximate (pinned) and exact references.
5. Record requester peak RSS, keygen time, key bytes, per-token upload and download bytes, round-trip latency,
   evaluator compute per step, and the dual-evaluation cost separately (second evaluator's compute and upload,
   requester hashing time).
6. Negative runs on the same machines: `--misbehave ask-decrypt`, `tamper`, `wrong-seq`; each must end with zero
   decryptions.

## 7. Review requirement

Before anything advertises `requester-only`, an independent cryptographer reviews: the parameters (lattice
estimator on this exact N, Q, P and secret distribution; the prototype's 128-bit claim is only a comparison with
the commonly cited 1772-bit ceiling), the protocol (admission and session binding, key-epoch reuse, IND-CPA-D
exposure through published outputs, decrypt-after-verify ordering, behaviour on approximation failure, timing
leakage), the implementation (Lattigo deserialization of untrusted ciphertexts and keys, memory handling, side
channels on the requester), and the Reploid integration (no readable fallback, state and log separation).
Doppler's availability gate stays closed and Reploid keeps `CONFIDENTIAL_EXECUTION_UNAVAILABLE` until that review
and the two-machine proof are both done.
