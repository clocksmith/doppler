# Doppler Goals

## Mission & Thesis

Doppler is an AI-native model release foundry and evidence-backed WebGPU runtime for JavaScript.

**Portable local inference is ordinary software, not an opaque cloud service.**

Modern applications should execute, inspect, and package machine learning models as easily and reliably as any standard JavaScript dependency. Doppler eliminates the complexity of shipping client-side AI by delivering self-contained, cryptographically signed model capsules, deterministic quantization, and high-performance WebGPU kernels directly in browsers and desktop runtimes without cloud dependencies.

## Intended Beneficiaries

1. **Client & Web Application Developers**: Engineers deploying on-device generative text, reranking, and embeddings into web and Electron applications without hosting costs or privacy risks.
2. **Local Autonomous Systems (Reploid)**: Goal-directed agent runtimes requiring dependable, local token generation, embedding lookup, and verifiable execution receipts.
3. **Enterprise Privacy & Security Teams**: Organizations mandating strict zero-egress inference where prompt bytes and generated tokens never leave the host device.

## Desired Outcomes

1. **Independent Standalone Adoption**: Maintainers voluntarily retain Doppler-backed models because they are faster, simpler to ship, and more dependable than alternatives. Free adoption counts as full technical success.
2. **Signed Capsule Integrity**: Deliver immutable, content-addressed model packages with reproducible quantization, verified tokenizers, and execution graphs.
3. **Fail-Closed Execution**: Unsupported hardware capabilities, missing WebGPU extensions, or corrupt weights fail closed with explicit diagnostic receipts rather than falling back to unverified CPU paths.
4. **First-Class Observability**: Provide full token-level inspection—including surprisal, alternative candidate probabilities, and kernel execution timing—directly to application runtimes.

## Operating Loops

1. **Inference & Telemetry Loop**:
   ```text
   Load Capsule -> Verify Identity -> Allocate WebGPU Buffers -> Execute Kernels -> Stream Tokens & Evidence Receipts
   ```
2. **Model Qualification Loop**:
   ```text
   Source Weights -> Quantization & Graph Compilation -> Accuracy/Oracle Parity Verification -> Generate Signed Capsule & Release Receipt
   ```

## Strategic Constraints

- Provider neutrality: Doppler operates on standard WebGPU across browsers and runtimes without proprietary hardware lock-in.
- Standalone sufficiency: Doppler must succeed as a self-contained open library; revenue, partner dependencies, or multi-repo integrations do not gate technical completion.
- Deterministic verification: Evidence receipts must bind exact model hashes, kernel configurations, and execution environments.

## Explicit Exclusions

Doppler does not build end-user application workflows, universal unverified model catalogs, cloud-hosted API proxy layers, or unbacked performance claims.

---

Links:
- Strategic intent links to local invariants in [INTENT.md](INTENT.md).
- Technical boundaries and owned authority are chartered in [CATSCAN.md](CATSCAN.md).
