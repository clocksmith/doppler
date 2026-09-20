// Application-side acceptance. The caller supplies only the installed public
// host API; no pipeline, device singleton, or internal package path is used.
export async function runInstalledCapabilityLifecycle(host, descriptor, hooks, policy) {
  const require = (condition, message) => { if (!condition) throw new Error(message); };
  require(Number.isSafeInteger(policy.repeatRuns) && policy.repeatRuns >= 2, 'At least two repeated requests are required.');
  require(['pre-aborted', 'after-partial'].includes(policy.cancellation), 'Declare the cancellation boundary.');
  const sessions = [];
  const observations = [];
  const checks = [];
  const openOptions = { ...descriptor.openOptions, persistReleaseCheckpoint: hooks.persistReleaseCheckpoint,
    observer: { observe: hooks.onProgress } };
  const request = () => ({ ...descriptor.request, ...(policy.adapter ? { adapterSet: [policy.adapter.entry] } : {}), limits: { ...descriptor.request.limits,
    deadlineAt: Date.now() + descriptor.maxDurationMs } });
  const adapterArtifactStore = policy.adapter ? { async readArtifact(artifact) {
    require(artifact.hash === policy.adapter.entry.artifact.hash, 'Unexpected adapter artifact');
    const response = await fetch(new URL(policy.adapter.weightsPath, descriptor.capsuleUrl));
    require(response.ok, `Adapter fetch failed: ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  } } : undefined;
  async function open(options = openOptions) {
    const session = await host.openCapsule(descriptor.capsuleUrl, options);
    sessions.push(session);
    return session;
  }
  async function execute(session, phase, baseOnly = false) {
    const input = request();
    if (baseOnly) input.adapterSet = [];
    const accumulator = input.schema === 'doppler.capsule-operation-request/v2'
      ? host.createCapsuleStreamAccumulator(input) : null;
    let completed;
    let partials = 0;
    for await (const event of session.executeOperation(input, { adapterArtifactStore })) {
      accumulator?.accept(event);
      if (event.status === 'partial') partials++;
      if (event.status === 'completed') completed = event;
    }
    require(completed, `${phase}: operation ended without completion.`);
    if (accumulator) completed = accumulator.finish();
    if (policy.adapter && !baseOnly) {
      const receipts = completed.receipt?.adapterReceipts;
      require(receipts?.length === 1 && receipts[0].sourceDigest === policy.adapter.entry.artifact.hash
        && receipts[0].identity === policy.adapter.entry.identity
        && receipts[0].runtimeIdentity?.schema === 'doppler.lora-execution-identity/v1'
        && /^sha256:[a-f0-9]{64}$/.test(receipts[0].runtimeIdentity.digest),
      'Completion must identify the physically loaded adapter tensors.');
    }
    observations.push({ phase, completed, partials });
    await hooks.onObservation?.({ phase, completed, partials });
    await hooks.onProgress?.({ type: 'acceptance-operation-completed', phase });
    return completed;
  }
  let failure;
  try {
    const first = await open();
    const manifest = first.manifest;
    const modelManifest = { modelType: manifest.modelType, architecture: manifest.architecture,
      artifactIdentity: manifest.artifactIdentity, inference: { supportsEmbedding: manifest.inference.supportsEmbedding,
        output: manifest.inference.output, rerank: manifest.inference.rerank } };
    for (let i = 0; i < policy.repeatRuns; i++) await execute(first, `repeat-${i}`);
    if (policy.adapter) {
      await execute(first, 'base-after-adapter-unload', true);
      let rejected;
      try {
        for await (const event of first.executeOperation(request(), { adapterArtifactStore: {
          async readArtifact() { throw new Error('Acceptance adapter preparation read failure.'); },
        } })) { void event; }
      } catch (error) { rejected = error; }
      require(rejected?.message.includes('Acceptance adapter preparation read failure'), 'Adapter preparation must reject.');
      checks.push({ id: 'failed-adapter-preparation', passed: true, error: rejected.message });
      await execute(first, 'base-after-failed-adapter-preparation', true);
    }

    const preparation = new AbortController();
    let preparationObserved = false;
    let preparationFailure;
    try {
      await open({ ...openOptions, signal: preparation.signal, observer: { observe(event) {
        hooks.onProgress?.(event);
        if (event.type === 'target-selected') {
          preparationObserved = true;
          preparation.abort(new Error('Acceptance cancels preparation after target selection.'));
        }
      } } });
    } catch (error) { preparationFailure = error; }
    require(preparationObserved && preparation.signal.aborted && preparationFailure
      && /abort|cancel/i.test(preparationFailure.message), 'Preparation must reject at its declared boundary.');
    checks.push({ id: 'cancelled-model-preparation', passed: true,
      error: { name: preparationFailure.name, message: preparationFailure.message } });

    const cancellation = new AbortController();
    let partials = 0;
    let completions = 0;
    let rejection;
    if (policy.cancellation === 'pre-aborted') cancellation.abort(new Error('Acceptance cancellation before execution.'));
    try {
      for await (const event of first.executeOperation(request(), { signal: cancellation.signal, adapterArtifactStore })) {
        if (event.status === 'completed') completions++;
        if (event.status === 'partial') {
          partials++;
          cancellation.abort(new Error('Acceptance cancellation after partial output.'));
        }
      }
    } catch (error) { rejection = error; }
    require(rejection && /abort|cancel/i.test(rejection.message) && completions === 0,
      'Cancelled operation must reject without accepting a completion.');
    require(policy.cancellation !== 'after-partial' || partials === 1, 'Expected exactly one partial before cancellation.');
    checks.push({ id: 'cancellation', passed: true, boundary: policy.cancellation, partials, completions,
      error: { name: rejection.name, message: rejection.message } });
    await execute(first, 'after-cancellation-and-failed-preparation');
    if (policy.adapter) await execute(first, 'base-after-adapter-cancellation', true);

    const second = await open();
    await first.close();
    require(first.closed && !second.closed, 'Closing the first session must not close the second.');
    await execute(second, 'second-after-first-close');
    checks.push({ id: 'second-session-survives-first-close', passed: true });
    if (policy.reopenCycles != null) {
      require(Number.isSafeInteger(policy.reopenCycles) && policy.reopenCycles > 0,
        'Reopen cycles must be a positive integer.');
      await second.close();
      await hooks.onProgress?.({ type: 'acceptance-cycle-closed', cycle: 0 });
      for (let cycle = 1; cycle <= policy.reopenCycles; cycle++) {
        const reopened = await open();
        await execute(reopened, `reopened-${cycle}`);
        await reopened.close();
        await hooks.onProgress?.({ type: 'acceptance-cycle-closed', cycle });
      }
      checks.push({ id: 'repeated-open-execute-close', passed: true, cycles: policy.reopenCycles });
    }
    return { completed: observations[0].completed, partials: observations[0].partials,
      lifecycle: { checks, observations, modelManifest } };
  } catch (error) { failure = error; throw error; }
  finally {
    // Iterators above are drained/returned before close: a paused iterator must
    // not hold its operation while the consumer awaits shutdown.
    const cleanup = await Promise.allSettled(sessions.map(session => session.close()));
    const errors = cleanup.filter(row => row.status === 'rejected').map(row => row.reason);
    if (errors.length) throw new AggregateError(failure ? [failure, ...errors] : errors,
      failure?.message ?? 'Installed consumer cleanup failed.', { cause: failure });
  }
}
