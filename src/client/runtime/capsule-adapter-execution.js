import { CAPSULE_ADAPTER_POLICY, resolveCapsuleAdapterSet } from '../../config/capsule-adapters.js';
import { hashCapsuleObservation, normalizeCapsuleObservation } from '../../config/capsule-operation.js';
import { freezeCapsuleV2 } from '../../config/capsule-v2.js';
import { sha256Hex } from '../../formats/sha256.js';
import { assertBundledResolutionNotRevoked } from '../../config/revocation-policy.js';

const assert = (ok, message) => { if (!ok) throw new Error(`Capsule adapter: ${message}`); };

// Runs inside the operation executor's exclusive slot. Bytes and math retain their existing owners.
export function createCapsuleAdapterExecution({ program, capsule, targetPlan }) {
  return async function prepare(request, { adapterArtifactStore, signal }) {
    const entries = resolveCapsuleAdapterSet(request.adapterSet === undefined ? CAPSULE_ADAPTER_POLICY.legacyAdapterSet : request.adapterSet,
      { capsule, targetPlan, operation: request.operation.name });
    assert(!program.getActiveAdapterIdentity?.(), 'unexpected adapter already active');
    if (!entries.length) return null;
    assert(typeof adapterArtifactStore?.readArtifact === 'function' && typeof program.loadAdapter === 'function'
      && typeof program.unloadAdapter === 'function' && typeof program.getActiveAdapterIdentity === 'function', 'adapter execution ports required');
    const receipts = [];
    let active = false;
    const check = async () => {
      signal.throwIfAborted();
      for (const [index, entry] of entries.entries()) {
        const identity = program.getActiveAdapterIdentity();
        assert(identity?.digest === receipts[index]?.runtimeIdentity.digest, 'active adapter changed during execution');
        await assertBundledResolutionNotRevoked({ adapterId: entry.manifest.id, adapterDigest: entry.artifact.hash });
        await assertBundledResolutionNotRevoked({ adapterId: entry.manifest.id, adapterDigest: identity.digest });
      }
      signal.throwIfAborted();
    };
    const close = async () => {
      if (!active) return;
      try { await program.unloadAdapter(); }
      finally { active = false; await program.reset?.(); }
      assert(!program.getActiveAdapterIdentity(), 'adapter remained active after unloading');
    };
    try {
      for (const entry of entries) {
        signal.throwIfAborted();
        await assertBundledResolutionNotRevoked({ adapterId: entry.manifest.id, adapterDigest: entry.artifact.hash });
        const received = await adapterArtifactStore.readArtifact(entry.artifact);
        signal.throwIfAborted();
        assert(received instanceof Uint8Array && received.byteLength === entry.artifact.sizeBytes, 'adapter size mismatch');
        const bytes = Uint8Array.from(received);
        assert(`sha256:${sha256Hex(bytes)}` === entry.artifact.hash, 'adapter artifact corruption');
        await program.reset?.();
        active = true; // A partially failed loader must also unload.
        await program.loadAdapter(entry.manifest, { bytes, signal });
        signal.throwIfAborted();
        const runtimeIdentity = freezeCapsuleV2(normalizeCapsuleObservation(program.getActiveAdapterIdentity()));
        assert(runtimeIdentity?.schema === 'doppler.lora-execution-identity/v1'
          && /^sha256:[a-f0-9]{64}$/.test(runtimeIdentity.digest), 'exact loaded tensor identity required');
        receipts.push(freezeCapsuleV2({ identity: entry.identity, requestDigest: hashCapsuleObservation(entry),
          sourceDigest: entry.artifact.hash, runtimeIdentity }));
      }
      await check();
      return { receiptFields: { adapterReceipts: receipts }, check, close };
    } catch (error) {
      try { await close(); } catch (cleanup) { throw new AggregateError([error, cleanup], error.message, { cause: error }); }
      throw error;
    }
  };
}
