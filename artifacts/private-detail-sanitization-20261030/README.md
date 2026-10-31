# Private detail sanitization

On 2026-10-30 the current tree of this public repository was cleaned of the
operator's home-directory paths, removable-media user directory, fleet host
names, a tailnet address, an SSH login and an author email. Git history before
this change still contains the originals. The earlier
[machine address redaction](../address-redaction-20261013/README.md) covered
connection addresses only.

- Home directories became `$HOME`; the Linux media mount became `/media/$USER/`.
- Hosts use the existing aliases `amd-host`, `intel-host` and `mac-host`, plus
  `windows-host`. Report and artifact folders named after a host were renamed
  to the alias.
- Code that defaulted to the operator's media directory now derives it from the
  current user.

## Evidence bytes and digests

Sanitizing evidence changes its bytes, so every digest in the repository that
named a changed file or archive member was recomputed. The rebinding ran
bottom-up to a fixpoint: scrubbed files, then the archives holding them, then
each manifest, index, receipt or `SHA256SUMS` that cites them. Byte counts next
to a rebound digest were updated as well.

[`manifest.json`](manifest.json) records, per changed file, the sha256 of the
original and sanitized bytes, the replacement counts by category, and the
archive members that changed. `digestRebinds` lists every rebound digest with
its field. `runtimeOutputDigests` names the rebound digests whose field covers a
captured runtime output such as a GPU capture or log; those digests now describe
the sanitized capture, not the bytes the runtime originally wrote. JSON evidence
with rebound digests carries `"pathSanitized": true` and points here.

Only file-content digests changed. Signatures, git revisions, shard-set and
conversion-config digests and model identities were left alone. The maintainers
keep the exact original bytes outside every repository; their sha256 values are
the `originalSha256` fields in the manifest.

## Left unchanged

`unsanitizedExceptions` in the manifest lists files that still contain a path,
grouped by reason:

- `model-manifest-identity`: RDRR model manifests. Their sha256 is the model
  artifact identity that receipts cite.
- `npm-package`: package tarballs byte-identical to the copies published on the
  npm registry, and archives that contain them.
- `signed` and `public-download-digest`: files whose digest is bound by an
  Ed25519-signed capsule or release event, or by a recorded public download
  digest.
- `signed-release-event-text`: two sentences inside signed capsule release
  events still name the Intel host by its original name, and files that quote
  those events repeat them. Changing them would invalidate the signatures.

Archives whose only private detail was the tar owner name were not rewritten.
Tokenizer vocabularies contain ordinary words that match host names; those are
not machine references.
