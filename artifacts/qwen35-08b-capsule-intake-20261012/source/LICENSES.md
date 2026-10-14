# Licenses for these source files

The repository's MIT license does not apply to these files. All of them are
under the Apache License 2.0, and [LICENSE](LICENSE) is the upstream copy
from `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`,
byte-identical (sha256 `bbedc3fda3305820b977…`, "Copyright 2026 Alibaba
Cloud").

| Files | Origin |
|---|---|
| `config.json` | Upstream checkpoint file, unmodified. |
| `header.bin`, `headers.json`, `header-binding.json` | Derived: safetensors headers extracted from the upstream checkpoint. |
| `modeling_qwen3_5.py.txt` | Hugging Face Transformers `modeling_qwen3_5.py` (Apache-2.0), saved with a `.txt` suffix and keeping its own header. |
| `reference-semantics.json` | Derived: Doppler's record of the reference semantics. |
