
// Canonical operator identity for operator-level differential debugging.
//
// Builds deterministic opIds from (section, layerIdx, stageName).
// The same model, config, and prompt produce the same opId sequence
// across backends and devices.
//
// opId format:
//   layer.{N}.{stage}     — per-layer operators (e.g., layer.12.attn.q_proj)
//   {stage}               — global operators (e.g., embed.out, logits.out)
//
// This is the single canonical source for operator identity.
// Do not create parallel naming systems in debug hooks.

import { getOperatorClass, canonicalizeProbeStage } from './stage-names.js';

// ============================================================================
// opId Construction
// ============================================================================

export function buildOpId(stageName, layerIdx) {
  if (layerIdx !== undefined && layerIdx !== null && Number.isFinite(layerIdx)) {
    return `layer.${layerIdx}.${stageName}`;
  }
  return stageName;
}

export function buildOpIdFromProbeStage(probeStageName, layerIdx) {
  const canonical = canonicalizeProbeStage(probeStageName);
  if (!canonical) {
    throw new Error(`[OperatorIdentity] Unknown probe stage name: "${probeStageName}".`);
  }
  return buildOpId(canonical, layerIdx);
}

export function buildOpIdFromExecutionStep(resolvedStep) {
  const section = resolvedStep.section;
  const op = resolvedStep.op;
  const layerIdx = resolvedStep.layers?.[0] ?? null;

  const stageName = EXECUTION_OP_TO_STAGE[op] ?? `exec.${op}`;
  return buildOpId(stageName, layerIdx);
}

// ============================================================================
// Execution-v1 op → stage mapping
// ============================================================================

const EXECUTION_OP_TO_STAGE = Object.freeze({
  embed: 'embed.out',
  rmsnorm: 'attn.post_input_norm',
  attention: 'attn.out',
  ffn: 'ffn.out',
  residual_add: 'residual.add',
  cast: 'cast',
  final_norm: 'final_norm.out',
  logits: 'logits.out',
  noop: 'noop',
  save: 'save',
  load: 'load',
  conv: 'conv.out_proj',
  moe: 'moe.expert',
});

// ============================================================================
// Operator Metadata
// ============================================================================

export function buildOperatorMeta(stageName, options) {
  const opId = buildOpId(stageName, options.layerIdx);
  const operatorClass = getOperatorClass(stageName);

  return {
    opId,
    stageName,
    operatorClass,
    phase: options.phase ?? null,
    layerIdx: options.layerIdx ?? null,
    tokenIndex: options.tokenIndex ?? null,
    dtype: options.dtype ?? null,
    quantizationMode: options.quantizationMode ?? null,
    shapeSignature: options.shapeSignature ?? null,
  };
}

// ============================================================================
// Sequence Tracking
// ============================================================================

export class OperatorSequence {
  constructor() {
    this._ops = [];
    this._index = 0;
  }

  record(opMeta) {
    const entry = {
      ...opMeta,
      sequenceIndex: this._index++,
    };
    this._ops.push(entry);
    return entry;
  }

  get length() {
    return this._ops.length;
  }

  getOps() {
    return this._ops.slice();
  }

  getOpById(opId) {
    return this._ops.find((op) => op.opId === opId) ?? null;
  }

  getOpsByLayer(layerIdx) {
    return this._ops.filter((op) => op.layerIdx === layerIdx);
  }

  getOpsByClass(operatorClass) {
    return this._ops.filter((op) => op.operatorClass === operatorClass);
  }

  clear() {
    this._ops = [];
    this._index = 0;
  }

  toJSON() {
    return this._ops;
  }
}
