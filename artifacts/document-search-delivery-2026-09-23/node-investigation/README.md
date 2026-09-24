# Node follow-up investigation

Both individual models passed their retained source references on Node 22.22.1,
Dawn `webgpu` 0.4.0, and the same AMD hardware using the installed 0.6.2 archive.
The [summary](summary.json) verifies every observed model shader against the
retained Capsule source hash; adapter probes match the installed provider runtime.
The [retained probe](retained-node-probe.js) records the explicit local input paths.

An initial raw-loader experiment used current shaders. The evaluation builder
then rejected `rope_precompute.wgsl` because it differed from the manifest's
pinned digest. Those initial observations do not qualify the declared closure.
The retained model receipts here are the subsequent executions using the exact
Capsule shaders through the loader's explicit shader-base-path port.

No signed Node model release or installed Node search application is qualified.
These runs loaded the models individually; they do not establish simultaneous
residency, cancellation, persistence, or recovery for the application. Their raw
model loading timings omit Capsule verification and are not product startup claims.

Next: use the retained Program Bundle with Forge's existing additional-surface
qualification path, create separate signed releases, and run the shared search
application and unchanged corpus against its installed Node assets. The browser
release and identities remain unchanged. Electron follows that application check.
