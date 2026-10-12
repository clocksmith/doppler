# Post-run report-validator corrections

The original pre-run producer rejected the completed final 0.6.28 browser split capture. Its first assertion compared two different hash domains: the installed initial-execution-identity helper sorts keys lexicographically, while the artifact canonical helper sorts with `localeCompare`. The observed resolved execution graph is exactly equal to the frozen candidate graph. The corrected producer retains recomputation of the entire observed identity with the exact installed archive helper and compares the observed graph object directly against the frozen graph.

The next attempt reached an independently incorrect settlement assertion: the actual device-memory snapshot has no `lost` field. The frozen observer records `deviceEvents`, `observationErrors`, validation errors and completed queue barriers. The corrected producer requires empty event/error arrays, positive confirmed barriers, no retained/active/deferred owners and a safe nonnegative device-buffer peak within the unchanged 1,420,000,000-byte limit.

`producer-before.js` preserves the original producer. `producer-hash-only.js` preserves the exact intermediate producer that exposed the second mismatch. Both physical rejection logs and failing-before/passing-after CPU logs are retained. The final 51 tests include missing/changed graph, missing device observations, observed device loss, observation failure and unconfirmed settlement rejection.

The original pre-run expectation files are unchanged. The `direction-*-reevaluation.json` receipts bind the intermediate correction; `direction-*-reevaluation-v2.json` bind the final correction. Root independently reviewed both changes after execution. These are explicit post-run validator reviews, not rewritten pre-run acceptance. The final reports bind the original producer and expectations, corrected producer, exact raw capture and unchanged independent reference. No model bytes, runtime archive, graph, logits, tolerance or evaluator changed, and no GPU rerun was needed.

Observed identity graph hash: `sha256:1be75c19e08b6c9fe3fb94af624c070bdc940e7d32582c704d56b2775a387596`.
Artifact canonical graph hash: `sha256:ae98737d0a894848eaffd3a87c6cae412bedddb949a8117062631455a8311e14`.

Validation: `node --test artifacts/signed-resident-producer-20261012/produce-physical-reports.test.js` (51 passed). Both final CLI conversions also passed `assertResidentQualification` imported from the exact frozen archive.
