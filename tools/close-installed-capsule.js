// Installed probes own the model session, device singleton, and provider session.
// Every owner is closed even when an earlier close fails.
export async function closeInstalledCapsule({ closeSession, destroyDevice, releaseProvider }) {
  const errors = [];
  for (const close of [closeSession, destroyDevice, releaseProvider]) {
    try { await close(); } catch (error) { errors.push(error.message); }
  }
  return { passed: errors.length === 0, errors };
}
