# Doppler Support & Compatibility Card: Reploid (Qwen 3.5 0.8B)

## Overview

| Field | Value |
| :--- | :--- |
| **Integration ID** | `reploid-local-generation` |
| **Application** | [Reploid](https://github.com/clocksmith/reploid) (`clocksmith/reploid`) |
| **Application Revision** | `cf8a1e7bb5b2b8059e30399bb2db8f5e41c523e9` |
| **Workload** | Generation (Local Biocuration & Contradiction Resolution) |
| **Logical Model** | `qwen-3-5-0-8b-q4k-ehaf16` |
| **Artifact Variant ID** | `sha256:b0e11d284d95b1e5f3f6819f978bc4b9a353a73f18054d137aaacf6c4de7cd56` |
| **Execution ID** | `sha256:59eaedb61ffd0089c2f4e0c933d9ad2be74a6eceff2c03695b1fb5183933bef1` |
| **Qualification Status** | **Product-Supported** |
| **Qualified At** | `2026-08-15T12:00:00.000Z` |
| **Expires At** | `2026-11-15T12:00:00.000Z` |

---

## Machine-Verifiable Qualification Receipts

All qualification claims are digest-bound to immutable receipts under [`docs/status/reploid-generation/`](../status/reploid-generation/):

1. **Owner Confirmation:** [`owner-confirmation.json`](../status/reploid-generation/owner-confirmation.json) (`doppler.product-integration-owner-confirmation/v1`)
2. **Execution Identity:** [`identity.json`](../status/reploid-generation/identity.json) (`doppler_provider_receipt_v1`)
3. **Install to First Verified Output:** [`installToFirstVerifiedOutput.json`](../status/reploid-generation/installToFirstVerifiedOutput.json) (2,399.9ms vs 5,000.0ms budget)
4. **Source Task Quality Retention:** [`sourceTaskQualityRetention.json`](../status/reploid-generation/sourceTaskQualityRetention.json) (98.5% quality retention vs 95.0% threshold)
5. **Reliability:** [`reliability.json`](../status/reploid-generation/reliability.json) (100% success rate across 100 trials, 0 crashes, 0 OOMs, 0 device losses)
6. **Memory Budget:** [`memory.json`](../status/reploid-generation/memory.json) (Peak VRAM: 1.54 GB vs 2.14 GB ceiling with fail-closed preflight)
7. **Cold / Warm Latency:** [`coldWarmResponse.json`](../status/reploid-generation/coldWarmResponse.json) (Cold p95: 2,399.9ms, Warm p95: 450.0ms)
8. **Browser Hardware Qualification:** [`browserHardwareQualification.json`](../status/reploid-generation/browserHardwareQualification.json) (Apple Silicon M3 Metal WebGPU)
9. **Incumbent Control:** [`incumbentControl.json`](../status/reploid-generation/incumbentControl.json) (Matched against CPU ONNX baseline)
10. **Upgrade Requalification:** [`upgradeRequalification.json`](../status/reploid-generation/upgradeRequalification.json) (v0.5.0 -> v0.5.1 migration passed)
11. **Rollback & Revocation:** [`rollbackRevocation.json`](../status/reploid-generation/rollbackRevocation.json) (Rollback to v0.5.0 passed with observed revocation)
12. **Promotion Evidence:** [`promotion.json`](../status/reploid-generation/promotion.json) (`doppler.product-integration-promotion-evidence/v1`)

---

## One-Line JavaScript Integration

```javascript
import { dr } from 'doppler-gpu';

// Preflight & load in-process WebGPU model
const model = await dr.open('qwen-3-5-0-8b-q4k-ehaf16', {
  preflightMemory: true,
  maxVramBytes: 2 * 1024 * 1024 * 1024,
});

// Stream generation tokens
for await (const token of model.chat([
  { role: 'system', content: 'You are an autonomous biocuration assistant.' },
  { role: 'user', content: 'Summarize contradiction findings for UniProt:P04637.' },
])) {
  process.stdout.write(token);
}
```
