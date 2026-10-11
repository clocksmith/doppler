# Signed selective resident acquisition

Component: Doppler Capsule runtime, storage and repository tooling.
Intent: preserved. Boundary effects: the signed Capsule program may explicitly
reference an authenticated piece index; storage owns byte authentication and
host adapters continue to own acquisition. No model mathematics changed here.

The optional `program.pieceIndexArtifactId` references an artifact with role
`piece-index`, using the existing `doppler.verified-pieces/v1` bytes. V2 and V3
signatures cover the new reference and artifact. Older consumers reject this
additive role; no records are emitted or activated automatically. Whole-artifact
verification remains unchanged for ordinary sessions and residents without the
reference. The existing index producer verifies complete source shard hashes;
a regression proves failure cannot emit or overwrite an accepted index.

Selective opening requires the existing exact signed resident qualification,
assignment authorization, accepted target and initial execution identity. The
index and manifest are fully verified before reading complete authenticated
weight pieces. Mismatched identity, geometry, aliases, altered bytes and host
acquisition limits reject. Host-provided buffers and metadata are copied before
asynchronous hashing. Both adapters share one dependency-free piece reader.
The injected runtime closure grows from 74 to 76 modules, not the intermediate
185-module host-storage closure. The installed checker now rejects that dependency.

Receipts distinguish `resident-pieces` from whole-closure verification. Actual
ranges identify complete verified pieces; the immutable session observation is
explicitly `resident-opening`, not final execution metrics. Partial validation
is never stored as a full-closure cache entry. Close cancels and awaits owned
read wrappers, including when an acquisition host ignores abort; this does not
assert that an arbitrary external promise has stopped. Session, storage and
program cleanup failures remain attributable.

The retained diagnostic archive is Doppler 0.6.27 with SHA-256
`307104de5f55e120280cccb4ed84b53170339250c60c3ceee8682ffe3149ed25`.
It includes the combined numerical and sampling source changes already present
in the workspace. It is **not** the standard 0.6.27 integration archive
`9280ead598b098f9f3ad49574a001e88c3ad4b1a433d58349e79ee0e9c01fccc`.
No delivery pin, production model metadata or physical qualification changed.

Validation:

- The broad repository sweep passed 903 files. The final shared-reader cleanup
  correction followed that sweep; all 14 affected Capsule/storage/resident and
  producer files passed afterward at unchanged assertions. An additional already
  running broader sweep subsequently passed 904 files, including the new producer
  case; the final affected rerun and exact installed snapshot independently bind
  the final cleanup correction.
- Source types, reviewed compute boundaries, architecture, runtime closure and
  package closure pass. The core has 76 files and 810,943 source bytes.
- The exact final archive passes detached installation, all 32 public export
  imports/declarations, browser-condition exports, HTTP/direct parity and the
  signed selective fixture. Its runtime and resident factory use installed
  public imports; fixture builders resolve installed files, never the checkout.
- Tests cover v2/v3 schema validation, one required piece without the unrelated
  piece, unqualified admission, mutation during hashing, malformed indexes,
  corruption, budget rejection, cleanup errors and noncooperative acquisition.

Failures are preserved: the initial full sweep exposed an ordinary-session
read-order regression, fixed before the passing broad sweep; the first installed
fixture imported an intentionally omitted internal hashing helper, corrected in
the fixture adapter without expanding the package. The generic close regression
failed with one active byte after close before the shared lifetime correction.
Earlier diagnostic package receipts are retained separately. Exact source and
artifact hashes are in [bindings.json](bindings.json).

These are synthetic contract and lifecycle results. No real model has gained
signed partition qualification from these tests. Physical signed resident
execution remains subject to independent numerical evidence and exact signed
partition-specific qualification records.
