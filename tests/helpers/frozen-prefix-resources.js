// Diagnostic settlement must preserve an earlier model/device failure and
// still attempt every owned cleanup step after an observation failure.
export async function settleFrozenPrefixResources({ steps, primaryFailure, recordFailure }) {
  const failures = [];
  for (const [phase, operation] of steps) {
    try { await operation(); }
    catch (error) {
      failures.push(error);
      try { await recordFailure({ phase, name: error.name, message: error.message }); }
      catch (recordError) { failures.push(recordError); }
    }
  }
  if (!primaryFailure && failures.length) {
    throw new AggregateError(failures, 'Diagnostic resource settlement failed');
  }
  return failures.length;
}
