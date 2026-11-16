# Repository tooling

Owner: `doppler.repository-tooling`, defined in [CATSCAN.md](CATSCAN.md).
Maintainers review changes to its checks, generators and evidence handling.
Shipped command implementations belong in `src/tooling/`; repository automation
belongs here. Runtime imports into `tools/` fail the architecture gate. The
explicitly packaged converter entrypoint remains a tool entrypoint, not a runtime
import exemption.

## Checks and tests

Start with `npm run check:fast`. It checks source style and ownership, changed
baseline style rules, and receipt retention without GPUs or model downloads.
Run affected behavior tests through `node tools/run-node-tests.js <test-files>`.
`tests/tooling/` contains checker and generator tests; a passing static gate is
separate from numerical, device and model qualification.

A new invariant detector needs a positive fixture and a negative fixture that
would pass if the detector were broken. Test behavior at its boundary rather
than requiring one source-code spelling. Guide detectors also register their
scope, limitations and fixture in `tools/policies/style-guide-rules.json`.
Reuse existing checks and generation owners before adding another npm script.

The check gate selects affected generated-file checks from
[check-gate-policy.json](policies/check-gate-policy.json) and the scripts' input
closures. Use the matching existing `:sync` command only when changing its
canonical input, then its `:check` command. `npm run kernels:sync` already
orders kernel generators. A blanket sync is inappropriate for frozen Capsules,
receipts and model artifacts: regeneration changes their identity.

`package-resource-policy.json` owns published runtime resources. Package exports
and bins own runtime reachability; `package-source-closure.js` derives the source
and kernel-reference npm ignore files from that graph. `package.json.files`
contains broad package roots, without a second runtime-profile exclusion list.
Run `package:closure:sync`, `package:closure:check`, and `public:boundaries:check`
when changing those inputs; the latter inspects the actual npm archive.

## Architecture debt

[source-architecture-policy.json](policies/source-architecture-policy.json)
owns source boundaries and size debt. Text-pipeline files have a blocking
750-line ceiling. Other inherited soft-limit files cannot grow against actual
baseline source counts without fresh external authorization. Reviews name their
owner, reviewer, date, extraction target and removal condition. Consumer sets
for cleanup decisions and dependency exceptions are checked for drift.

CI supplies `DOPPLER_POLICY_BASE_REF` from the push's previous commit or PR base;
a manual workflow uses the preceding commit. Local checks default to `HEAD`.
Use `npm run source:architecture:check -- --base <revision>` to inspect an
explicit comparison. Do not mint authorization values to make a check pass.

## Receipt retention

[artifact-retention.json](policies/artifact-retention.json) owns retention for
tracked `artifacts/` and `reports/` campaigns. Its exact inventories bind paths,
Git blob identities, file counts and byte counts. Existing signed and historical
evidence uses an explicit owner and retirement condition; temporary retention
classes may carry a `retainUntil` date, and expired dates fail. The checker never
deletes or rewrites evidence. Old byte identities do not become newly qualified
when source code changes.

Local outputs stay ignored. To retain a deliberate new receipt, inspect its
contents and incoming references, use `git add -f <receipt-files>`, then run:

```sh
node tools/check-artifact-retention.js --inventory reports/<campaign>
```

Review the proposed inventory and update the campaign's policy entry and
retention class. Run `npm run check:fast` before committing. Unregistered,
modified, missing, expired and stale records fail. Removing a campaign requires
retiring or replacing its references and bound artifacts; Git history preserves
the retired record. Large existing evidence remains in place until that
identity-preserving migration is explicitly carried out.
