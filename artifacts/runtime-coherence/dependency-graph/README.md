# Parsed dependency checks

All five dependency checks consume the shared TypeScript syntax parser: Capsule
runtime closure, npm source inclusion, browser imports, source architecture and
public boundaries. Graph nodes distinguish static/dynamic/type imports, resource
literals and unresolved dynamic expressions. This is repository tooling only.

Regression tests reject forbidden literal dynamic imports and missing browser
modules, ignore imports inside comments/inert strings, traverse cycles and JSON
resources, and reject undeclared or changed nonliteral imports in the closed
Capsule runtime. Its two Node built-in expressions are explicit source-bound
policy rows; other host/config-selected imports remain visible in the broader
source inventory and are not represented as statically resolved paths.

The runtime inventory now includes the existing revocation-registry JSON resource.
The browser view retains the existing declared Node bridge exclusions. Package
budgets now account for the prior WGSL contract module and cache repair: 1,790
files and 10,917,992 unpacked bytes, with the existing compression allowance.
The initial budget failure is retained; no runtime or model bytes changed here.

Three regression files and all five checks passed. Repacking reproduced exactly
SHA-256 `068b2bbfb3b0568bb54eface1fee61809a703a3ec8cf4d49b6ecb8879900d490`,
the archive already accepted through standalone and Reploid installed tests in
`../subgroups/`. No GPU rerun was needed for these tooling changes.

```sh
node tools/run-node-tests.js tests/tooling/javascript-dependency-graph.test.js tests/tooling/runtime-closure-graph.test.js tests/tooling/browser-import-graph.test.js
node tools/run-node-tests.js --scripts runtime:closure:check package:closure:check imports:check:browser source:architecture:check public:boundaries:check
```

Component: doppler.repository-tooling.
Intent: preserved.
Acceptance evidence: retained tests/checks and exact archive reproduction.
Boundary effects: additional resource/import visibility in existing checks;
no runtime, Reploid source, publication or deployment changes.
