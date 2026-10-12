# Archive8 frozen-source CPU unit sweep

All **953 selected unit files** completed successfully across the initial sweep
and focused environment reruns. **46 files explicitly skipped optional cases**;
`file-results.json` preserves every observed skip and its reason. The canonical
runner also excludes one pending file, listed in `summary.json`. This is CPU
contract evidence, not GPU acceptance, model quality, signed opening, or release
qualification.

The source is `1cb70cf923822481ac170113be5a68cb4d01ab4e`, exported with
`git archive` outside the live checkout. Archive SHA-256 is
`8ea1767fb61d9009dd257c4065a6acfdbaaf80fa733145898307ed35da0e7f20`.
All 1,956 shipped files match the exported source before and after execution;
the final tracked-source diff is empty. Installed dependency lock bytes also
remain unchanged. See `binding.json`, `archive-files.json`, and
`source-after.json`. No runtime or test source was changed.

The x128 command was:

```sh
taskset -c 0,1,2,3 node tools/run-node-tests.js --suite unit
```

Node was v22.22.1. The canonical runner executes one file at a time. The exact
environment in `binding.json` bounds CPU-library threads to four and points
`DOPPLER_NODE_WEBGPU_MODULE` at an intentionally absent owned path. Existing
tests therefore take their ordinary unavailable-provider skip paths rather than
occupying a GPU. Other skips require absent local models, captures, retained
training evidence, or explicit physical-test inputs; one is an existing deferred
TranslateGemma scope. These skips were not changed into passing assertions.

## Retained failures and disposition

`unit.log.gz` retains the original 943 successful files and ten failures:

- Eight files require Git metadata/history or a commit identity. The exported
  tree initially had none. A private Git object store uses a read-only object
  alternate to the existing repository, detached HEAD at the frozen commit, and
  its own index. No live checkout/index or registered worktree was changed.
- `paired-model-binding.test.js` imports two sibling Reploid reference files.
  Their exact committed bytes were exported separately; `sibling-binding.json`
  records that revision and both hashes.
- `node-provider-standalone.test.js` installs a synthetic GPU before testing
  default bootstrap. The deliberate unavailable-provider override superseded
  it. Its focused rerun removes only that override and exercises the synthetic
  fixture; it does not request a physical GPU.

`environment-rerun.log.gz` retains seven successful repairs and two further
environment failures. The training fixtures require a real `node_modules`
directory and create their own temporary Git repositories. Inherited `GIT_*`
variables incorrectly redirected those nested repositories to the outer private
metadata. The final adapter uses a snapshot-local `.git` pointer without inherited
Git variables, and an owned dependency directory whose entries link to the
unchanged installed dependencies. Both training fixtures then pass in
`nested-workspace-rerun.log.gz`. The provider rerun is retained separately.
Only affected files were rerun. `metadata-first-failure.json` also preserves the
initial incorrect assumption that the workspace submodule had a `.git/objects`
directory; the corrected adapter uses its actual common object directory.

`summary.json` records each trial rather than replacing the original failures.
The exported tree and complete raw logs remain on x128 at the paths recorded in
`binding.json`; compressed logs here preserve their exact content.
