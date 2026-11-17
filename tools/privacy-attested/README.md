# Attested light-client route (research)

A requester checks hardware evidence from a confidential guest, encrypts its request to the key that evidence names and
decrypts the answer. The requester holds no weights and runs no model operation. The guest runs the ordinary plaintext
runtime.

This is an internal research client and provider, not an available Doppler backend.
`assertExecutionBackendAvailable()` refuses attested descriptors, and the installed
[trust policy](trust-policy.json) admits no deployment. The [design and threat model](DESIGN.md) state what is and is
not defended.

## What has and has not run

- The verifier accepts two genuine AMD SEV-SNP reports through AMD's real certificate chain
  ([fixtures](fixtures/README.md)) and refuses them when anything is wrong.
- The complete protocol runs between two processes with a software stand-in for the hardware
  ([receipt](evidence/local-software-run.json)). That establishes no isolation.
- No confidential guest has been provisioned. The [guest report source](attester-configfs.js) has never run on
  hardware. No model has run through this route.

## Files

| File | Purpose |
| --- | --- |
| [client.js](client.js) | The requester: the six operations of Doppler's execution-backend extension, plus `run` |
| [provider.js](provider.js) | The guest side: key, report request, decryption, generation, encrypted stream |
| [evidence.js](evidence.js) | Appraisal order, trust-policy validation and the installed verifier set |
| [snp-report.js](snp-report.js), [snp-endorsement.js](snp-endorsement.js) | Report layout; AMD root pins and chip certificate |
| [session.js](session.js) | Report-data binding, key schedule, authenticated frames, padding |
| [wire.js](wire.js) | Untrusted message channels |
| [attester-configfs.js](attester-configfs.js) | Report source for a Linux SEV-SNP guest |
| [test-support.js](test-support.js), [provider-process.js](provider-process.js) | Software stand-in chip and provider, for tests and the local run |

## Run

```sh
node --test tools/privacy-attested/attested.test.js
node tools/privacy-attested/run-local.js --output=/tmp/attested-run.json
```

Both need only Node 22 and no network.
