# Machine address redaction

Tracked files in this repository named the operator's private machine
addresses. On 2026-10-13 every occurrence in the current tree was replaced with
a neutral alias. Git history before this commit still contains the originals.

- `amd-host`, `intel-host`, `mac-host`: the three machines used for physical
  runs. Connection addresses belong in local SSH configuration, not in this
  repository.
- `other-host`, `tailnet.invalid`, `tailnet-redacted`: any other occurrence.

[`manifest.json`](manifest.json) lists each changed file with the SHA-256 of
its original and redacted bytes. The redacted files are derived copies, not the
qualification records: hash bindings elsewhere in this repository refer to the
original bytes, which the operator retains privately. Plain text, gzip files
and tar archives were rewritten; only the addresses changed.

New evidence must use the aliases. Retained scripts that connect over SSH now
expect those aliases to resolve through local configuration.
