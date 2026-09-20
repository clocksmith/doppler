import { resetSequenceState } from '../../src/inference/pipelines/text/sequence-state.js';
import { matchesStopSequence } from '../../src/inference/pipelines/text/stopping.js';
import { getCachedPipeline } from '../../src/gpu/kernels/pipeline-cache.js';
import { getBufferPool } from '../../src/memory/buffer-pool.js';
import { generationRequestEvidence } from '../../src/inference/pipelines/text/generation-request.js';

// Compile-only negative contracts: removing a required field or confusing a
// request/model/device owner must make this check fail (unused expect-error).
// @ts-expect-error A string is not a sequence length.
resetSequenceState({ isGenerating: false, currentSeqLen: 4 }, '2');
// @ts-expect-error Cache state must declare the active length.
resetSequenceState({ isGenerating: false }, 0);
// @ts-expect-error Stop strings cannot be token IDs.
matchesStopSequence({ decode: () => '' }, [], 0, [4]);
// @ts-expect-error A pool is not a GPU device.
getCachedPipeline('matmul', 'f32', null, getBufferPool());
// @ts-expect-error Partial sampling knobs are not a resolved execution request.
generationRequestEvidence({ presencePenalty: 0.5 });

/** @param {import('../../src/storage/model-read-session.js').ModelReadSession} store */
function checkStore(store) {
  // @ts-expect-error Opened read handles cannot be rebound to another model.
  store.openModel('other');
  // @ts-expect-error Model identity is immutable.
  store.modelId = 'other';
  return store.readRange('weights.bin', 0, 4);
}
void checkStore;

/** @param {import('../../src/client/runtime/capsule-acquisition.js').CapsuleArtifactReader} reader
 * @param {import('../../src/config/capsule-v2.js').CapsuleV2Artifact} artifact */
function checkArtifactReader(reader, artifact) {
  // @ts-expect-error A stream must declare its acquisition bound.
  reader.streamArtifact?.(artifact, {});
  // @ts-expect-error Chunk limits are numbers, not strings.
  reader.streamArtifact?.(artifact, { maxChunkBytes: '1024' });
  // @ts-expect-error Chunk streams must produce bytes rather than strings.
  reader.streamArtifact = async function* () { yield 'untrusted text'; };
}
void checkArtifactReader;
