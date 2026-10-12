# Development unit sweep after paired integration

Component: `doppler.runtime-source.inference`
Intent: preserved. Boundary effects: kernel registry, dispatch validation and test fixtures.

`npm run test:unit` completed 946 test files: 943 passed and three failed. The full unfavorable transcript is retained in `unit-sweep.txt`.

All three failing files subsequently passed their focused reruns (`focused-rerun.txt`):

- `tests/config/kernel-ref-digests.test.js`: refresh the generated registry for newly registered WGSL; fixed in `88fcba2c`.
- `tests/gpu/matmul-dispatch-contract.test.js`: shared dispatch validation now covers paired reduction geometry and second-axis overflow; fixed in `f28a98f8`.
- `tests/inference/runtime-contract-regressions.test.js`: make the scalar fixture's intermediate precision explicit so it continues testing the missing-epsilon rejection rather than failing earlier on an unrelated absent field.

This was a development working-tree sweep while separately owned work continued. It is not a clean immutable-archive sweep, a whole-model numerical result, or distributed acceptance. Focused corrections do not rewrite the original three failures into a passing historical run. Physical tests in the suite used the reserved Mac GPU window; the window was released after the runner exited.
