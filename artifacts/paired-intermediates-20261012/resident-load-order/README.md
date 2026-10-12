# Resident admission must consume the resolved session

Archive `8ea1767fb61d9009dd257c4065a6acfdbaaf80fa733145898307ed35da0e7f20` passed the retained unsplit 55-prefix tests on AMD and Mac. Its first Mac/AMD resident split attempt failed during preparation, before any resident descriptor or numerical output, with `Resident partition requires explicit resolved intermediate precision.` This is not a failed numerical comparison.

The lifecycle called the strict resident validator before assigning `resolvedRuntimeSession`. The owning repair moves admission immediately after that assignment, still before weight loading. It changes no validator, requested precision, graph, weights, reference, tolerance, or GPU arithmetic.

`node tests/inference/partition-load-session-order.test.js` executes the real load lifecycle and real partition validator with CPU model-loading fixtures. Before the repair, the first supported scalar case fails at the same boundary. After it, scalar and paired cases reach weight loading with immutable session policy, while both crossed precision combinations fail before weight loading. `before.log` and `after.log` preserve those results. The signed paired-opening CPU fixture also passes.

This repair requires a new ordinary package and fresh numerical, resident, and signed qualification. The prior unsplit receipts do not qualify new package bytes. The retained physical preparation receipt remains pinned to archive 8.

Component: `doppler.runtime-source.inference.pipelines.text`. Intent: preserved. Boundary effects: resident admission now receives its required resolved session. No application pin change.

Archive9 is ordinary package SHA256 `8cae56b0748a5c66add985122d8bbf72b7492e2e0828f713f1b0d7425a59a4e5`, built from clean canonical commit `bddea7ef`. Its 1,974 packaged files match that exported source byte-for-byte. Intervening committed RD-CANON work is included; the numerical run uses the retained legacy profile and unchanged paired graph/model.

The AMD full55 replay passes the unchanged approved F64 reference: maximum error `9.520044947919359e-7`, RMS `7.555127322679271e-8`, zero failing prefixes or values above0.001. All57 outputs (55unique plus resident repeats) are byte-identical to the same-provider archive8 outputs. Device-buffer peak is1,132,260,560bytes under1.42billion; the controller completed and closed its owned browser. Raw output, independent comparison, and resource observations are retained. Matching Mac and physical split qualification remain pending at this checkpoint.
