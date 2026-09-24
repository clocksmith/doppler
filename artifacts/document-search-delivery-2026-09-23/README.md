# Browser search starter 0.1.0

The [distribution receipt](distribution.json) pins the publicly downloadable archive,
its installed executable assets, runtime archive, lockfile, and both signed Capsules.
Every executable asset matched the passing physical application after a separate
archive extraction and frozen installation. All 32 public model shards passed
size and hash verification; none has a missing source.

- [Public model-byte audit](public-source-audit.json)
- [Installed clean-consumer check](installed-check.json)
- [Physical application qualification](browser-qualification.json)
- [Retained reference corpus](reference-corpus.json)
- [Repository validation](repository-check.json) and [full output](check-green.txt)

The physical receipt covers Chrome 146.0.7680.177 on Linux, AMD Radeon 8060S
(RADV STRIX_HALO), with both models resident. Six reference queries matched online
and offline. Unchanged indexing, submitted-work cancellation, superseded queries,
interrupted saves, explicit closure, corruption repair, and device loss passed.
The server was stopped and browser networking disabled for reopening.

The profile filesystem was `tmpfs`. This proves browser-process restart and does
not establish reboot persistence, disk-backed timing, or a minimum memory size.
The initial disk-backed attempt failed on exhausted host storage before qualifying.
See the [starter guide](../../examples/document-search/README.md) for observed timings
and exact prerequisites. Independent adoption, Node execution, Electron, and Bun
are not established by this receipt.

All 863 repository test files passed, but `check:green` returned failure for five
existing layer-partition declaration/export/closure checks. They also affect the
pinned archive's unrelated public declarations. This is an application release,
not a new npm runtime release or a claim that repository validation is green.
Runtime-source ownership and behavior were not changed to waive those failures.

Maintenance scope: preserve immutable model and package identities, reproduce
reported installation/search/lifecycle defects against this configuration, retain
regression evidence, and publish changed executable assets only after fresh
installed qualification. New platforms require their own receipts. External
integration feedback and a second independently operated revision remain pending.

The [independent integration invitation](https://github.com/clocksmith/doppler/issues/13)
is public; no unrelated developer has yet accepted or completed it. The
[portability preflight](portability-preflight.json) confirms that the installed
runtime rejects Node execution for both existing browser-only model releases.
A Node runner requires newly qualified model releases, followed by installed
application acceptance; changing the host label would invalidate this evidence.
