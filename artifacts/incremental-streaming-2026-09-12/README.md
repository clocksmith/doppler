# Streaming and lifecycle checkpoint

V2 output events carry new tokens/text or one embedding item. The installed
candidate receipt identifies SHA-256
`f012daff99372165bd7721c9a985abe2e101eda19bf680de4479b48594b1c454`.
Standalone direct/HTTP and Reploid installed contract tests passed on those bytes.
That archive precedes the lifecycle repairs and latest upstream model-cache edit.
The current package audit records the updated payload; it is not a physical receipt.
Published npm 0.6.1 is a different archive, retained in the installed baseline record.

`transport.json` measures synthetic output processing with the actual tokenizer,
operation adapters, executor and consumer helper. V2 transfers 472285, 944349 and
1888477 bytes for 1024, 2048 and 4096 tokens. V1 transfers 27142194 bytes at 4096.
Logical payload counts and V8 allocation estimates have their stated limits;
none measures GPU inference speed. Native journal before/after logs separately
show quadratic cumulative writes replaced by approximately linear append traffic.

Shared-device failure injection found stale token delivery after loss, admission
of a program loaded after loss, reuse of a released slot after allocation failure,
and incomplete cleanup after a release hook throws. All five lifecycle tests pass
after the focused fixes, including independent sessions and draining pending work.
These are deterministic injected-resource tests, not physical-GPU qualification.

The broad local check passed 799/801 unit files before the lifecycle edit. Two
training tests rejected the worktree dependency symlink; both pass after restoring
a real dependency directory. Six focused regression files pass after the repairs.
Remote CI, final installed archive and affected physical runs remain pending.

Component: doppler.runtime-source.client, doppler.tests, doppler.repository-tooling.
Intent: preserved.
Acceptance evidence: retained transport, installed archive and lifecycle records.
Boundary effects: public v2 operation format and consumer helpers; existing resource ownership.
