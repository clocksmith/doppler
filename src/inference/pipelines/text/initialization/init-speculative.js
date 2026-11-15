import { SpeculativeDecoder } from "../../../speculative.js";
import { log } from "../../../../debug/index.js";

// ============================================================================
// Speculative Decoder Setup
// ============================================================================

// EXPERIMENTAL: Speculative decoding is parsed and initialized but the full
// verify-and-accept loop is not yet wired into the generation pipeline.
// Enabling this will create the decoder state but decoded tokens are not
// verified against the draft model. Do not rely on this for production use.
export function initSpeculativeDecoder(manifest, speculativeConfig) {
  if (!manifest.draftModel) return null;
  if (manifest.draftModel.numTokens == null) {
    throw new Error(`Manifest "${manifest.modelId}" is missing draftModel.numTokens.`);
  }

  log.warn(
    'Pipeline',
    `Speculative decoding enabled for "${manifest.modelId}" but this feature is experimental and not fully wired. ` +
    'The draft-verify-accept loop is incomplete. Generated output is from the base model only.'
  );

  return new SpeculativeDecoder({
    numDraftTokens: manifest.draftModel.numTokens,
    maxRejectionRetries: speculativeConfig.maxRejectionRetries,
    enableTreeDraft: speculativeConfig.enableTreeDraft,
    temperature: speculativeConfig.temperature,
    randomSeed: speculativeConfig.randomSeed,
  });
}
