# Public SEV-SNP evidence used by the tests

These are genuine AMD SEV-SNP attestation reports and chip certificates published as test data by two open-source
projects, both under the Apache License 2.0. They let the verifier be tested against real hardware signatures and AMD's
real certificate chain. None of them comes from a Doppler deployment.

| File | Source | SHA-256 |
| --- | --- | --- |
| `milan.report.bin` | [virtee/sev](https://github.com/virtee/sev/blob/c10a378624e1d69d32f79658b72685961ee7ab17/tests/certs_data/report_milan.hex), converted from hex to bytes | `120d77b213c8868dd42f160ccb0114f05336ec715f6d51070f534b33c7e03f3b` |
| `milan.vcek.der` | [virtee/sev](https://github.com/virtee/sev/blob/feaf142ec916c23e7658f088802921177ee290f9/tests/certs_data/vcek_milan.der) | `3bbfb6ee259f75a95d13168cfdf2e034181bb93c7c016825731cbe8ea16c95e1` |
| `milan-debug-policy.report.bin` | [google/go-sev-guest](https://github.com/google/go-sev-guest/blob/609f29a271138914d1e241d3bd45309eacb42d47/verify/testdata/attestation.bin) | `377e6241d3b373ab1df80c0f96978594e7e21f4797dd6ea95e2957e1c1e26060` |
| `milan-debug-policy.vcek.der` | [google/go-sev-guest](https://github.com/google/go-sev-guest/blob/a30aab2ec76981a7ab4d6909df57f2ca9051d2b4/verify/testdata/vcek.testcer) | `0d057f9b6e29a69eda9c0154b259567d291c1c08d73a11e9d31ace07c435b6d8` |

The second report was produced by a guest whose policy allows debugging. The verifier accepts its signature and then
refuses it for that reason, which is the behaviour the test checks.

`../roots/amd-milan.pem` and `../roots/amd-genoa.pem` are the ASK and ARK certificates served by AMD's key distribution
service (`https://kdsintf.amd.com/vcek/v1/{Milan,Genoa}/cert_chain`, fetched 2026-11-17). The trust policy pins the
SHA-256 of each certificate's public key, so replacing these files does not change what the client trusts.
