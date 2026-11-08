// Which declared vision architectures the runtime can execute faithfully.
//
// A manifest may declare a vision tower whose weights and config load but whose
// forward pass is not implemented to the reference. Such a model must not
// advertise vision capability and must not silently produce image tokens.
// Qwen3-VL is in that state: the shared ViT path applies no 2D rotary or
// learned position embedding, the merger never applies visual.merger.ln_q,
// and the deepstack features are never merged into the text layers.
export const VISION_ARCHITECTURE_SUPPORT = Object.freeze({
  gemma4: Object.freeze({ executable: true, reason: null }),
  glmocr: Object.freeze({ executable: true, reason: null }),
  qwen3vl: Object.freeze({
    executable: false,
    reason: 'the shared ViT path applies no 2D rotary or learned position embedding, '
      + 'the merger never applies visual.merger.ln_q, and deepstack features never reach the text layers',
  }),
});

/**
 * @param {unknown} visionArchitecture
 * @returns {{ architecture: string, declared: boolean, executable: boolean, reason: string | null }}
 */
export function resolveVisionArchitectureSupport(visionArchitecture) {
  const architecture = typeof visionArchitecture === 'string' ? visionArchitecture.trim() : '';
  if (!architecture || !Object.hasOwn(VISION_ARCHITECTURE_SUPPORT, architecture)) {
    return { architecture, declared: false, executable: false, reason: 'no vision encoder is implemented for this architecture' };
  }
  const entry = VISION_ARCHITECTURE_SUPPORT[/** @type {keyof typeof VISION_ARCHITECTURE_SUPPORT} */ (architecture)];
  return { architecture, declared: true, executable: entry.executable, reason: entry.reason };
}

/** @returns {string[]} */
export function executableVisionArchitectures() {
  return Object.entries(VISION_ARCHITECTURE_SUPPORT)
    .filter(([, entry]) => entry.executable)
    .map(([name]) => name);
}

// Pipeline vision capability from the manifest gate (image_token_id plus
// quantizationInfo.vision) and the support table. Capability means the runtime
// can execute the declared tower, not merely that the manifest declares one.
/**
 * @param {{ image_token_id?: unknown, quantizationInfo?: { vision?: unknown } | null } | null | undefined} manifest
 * @param {{ visionConfig?: { visionArchitecture?: unknown } | null } | null | undefined} modelConfig
 * @returns {{
 *   state: { visionCapable: boolean, imageTokenId: number | null, visionConfig: object | null, visionUnsupportedReason: string | null },
 *   info: string | null,
 *   warning: string | null,
 * }}
 */
export function resolveVisionCapability(manifest, modelConfig) {
  const imageTokenId = manifest?.image_token_id;
  const hasVisionQuant = manifest?.quantizationInfo?.vision != null;
  const absent = { visionCapable: false, imageTokenId: null, visionConfig: null, visionUnsupportedReason: null };
  if (!Number.isInteger(imageTokenId) || /** @type {number} */ (imageTokenId) <= 0 || !hasVisionQuant) {
    return { state: absent, info: null, warning: null };
  }
  const visionConfig = modelConfig?.visionConfig ?? null;
  if (!visionConfig) {
    throw new Error(
      `Manifest declares image_token_id=${imageTokenId} and quantizationInfo.vision ` +
      'but no vision_config was resolved. Check conversion config.'
    );
  }
  const support = resolveVisionArchitectureSupport(visionConfig.visionArchitecture);
  if (support.executable) {
    return {
      state: { visionCapable: true, imageTokenId: /** @type {number} */ (imageTokenId), visionConfig, visionUnsupportedReason: null },
      info: `Vision capable: imageTokenId=${imageTokenId}, architecture=${support.architecture}`,
      warning: null,
    };
  }
  const reason = `vision architecture "${support.architecture}" is declared by the manifest but not executable (${support.reason})`;
  return {
    state: { visionCapable: false, imageTokenId: /** @type {number} */ (imageTokenId), visionConfig, visionUnsupportedReason: reason },
    info: null,
    warning: `Vision unavailable: ${reason}.`,
  };
}

// Message for a vision entry point on a pipeline whose vision is unavailable:
// either the manifest declares none, or it declares one the runtime cannot run.
/**
 * @param {{ visionUnsupportedReason?: string | null } | null | undefined} state
 * @param {string} operation
 * @returns {string}
 */
export function describeVisionUnavailable(state, operation) {
  if (state?.visionUnsupportedReason) {
    return `Pipeline cannot run ${operation}: ${state.visionUnsupportedReason}.`;
  }
  return `Pipeline does not support ${operation} (no image_token_id in manifest).`;
}
