
// Operator execution record emission for operator-level differential debugging.
//
// Emits OperatorExecutionRecords at semantic operator boundaries.
// Records contain both semantic context (from Doppler) and placeholders
// for execution truth (filled by Doe/Fawn at runtime).
//
// The emitter accumulates a timeline of records for a single inference
// run. The timeline is the substrate for first-divergence detection.

import { buildOpId, buildOperatorMeta } from './operator-identity.js';
import { getOperatorClass } from './stage-names.js';

// ============================================================================
// Record Construction
// ============================================================================

export function createOperatorExecutionRecord(options) {
  const stageName = options.stageName;
  const opId = options.opId ?? buildOpId(stageName, options.layerIdx);
  const operatorClass = getOperatorClass(stageName);

  return {
    // Model identity
    modelHash: options.modelHash ?? null,
    adapterHash: options.adapterHash ?? null,
    runtimeConfigHash: options.runtimeConfigHash ?? null,
    executionPlanHash: options.executionPlanHash ?? null,

    // Semantic context (Doppler-owned)
    phase: options.phase ?? null,
    tokenIndex: options.tokenIndex ?? null,
    layerIndex: options.layerIdx ?? null,
    opId,
    opType: operatorClass,
    stageName,
    shapeSignature: options.shapeSignature ?? null,
    dtype: options.dtype ?? null,
    quantizationMode: options.quantizationMode ?? null,
    inputTensorIds: options.inputTensorIds ?? [],
    outputTensorIds: options.outputTensorIds ?? [],

    // Capture and drift policy (Doppler-owned)
    capturePolicy: options.capturePolicy ?? 'none',
    driftPolicyId: options.driftPolicyId ?? operatorClass,

    // Execution truth (Doe-owned, filled at runtime)
    kernelDigest: options.kernelDigest ?? null,
    wgslHash: options.wgslHash ?? null,
    pipelineHash: options.pipelineHash ?? null,
    backend: options.backend ?? null,
    adapterVendor: options.adapterVendor ?? null,
    adapterArchitecture: options.adapterArchitecture ?? null,
    driverVersion: options.driverVersion ?? null,
    workgroupGeometry: options.workgroupGeometry ?? null,
    dispatchCount: options.dispatchCount ?? null,
    timing: options.timing ?? null,
    captureArtifactIds: options.captureArtifactIds ?? [],
  };
}

// ============================================================================
// Operator Event Emitter
// ============================================================================

export class OperatorEventEmitter {
  constructor(options = {}) {
    this._records = [];
    this._modelHash = options.modelHash ?? null;
    this._adapterHash = options.adapterHash ?? null;
    this._runtimeConfigHash = options.runtimeConfigHash ?? null;
    this._executionPlanHash = options.executionPlanHash ?? null;
    this._enabled = options.enabled !== false;
    this._activeOp = null;
  }

  get enabled() {
    return this._enabled;
  }

  enable() {
    this._enabled = true;
  }

  disable() {
    this._enabled = false;
  }

  beginOp(stageName, options = {}) {
    if (!this._enabled) return null;

    const record = createOperatorExecutionRecord({
      ...options,
      stageName,
      modelHash: this._modelHash,
      adapterHash: this._adapterHash,
      runtimeConfigHash: this._runtimeConfigHash,
      executionPlanHash: this._executionPlanHash,
    });

    this._activeOp = {
      record,
      startTime: performance.now(),
    };

    return record.opId;
  }

  endOp(executionFacts) {
    if (!this._enabled || !this._activeOp) return null;

    const { record, startTime } = this._activeOp;
    const endTime = performance.now();

    if (executionFacts) {
      record.kernelDigest = executionFacts.kernelDigest ?? record.kernelDigest;
      record.wgslHash = executionFacts.wgslHash ?? record.wgslHash;
      record.pipelineHash = executionFacts.pipelineHash ?? record.pipelineHash;
      record.backend = executionFacts.backend ?? record.backend;
      record.adapterVendor = executionFacts.adapterVendor ?? record.adapterVendor;
      record.adapterArchitecture = executionFacts.adapterArchitecture ?? record.adapterArchitecture;
      record.driverVersion = executionFacts.driverVersion ?? record.driverVersion;
      record.workgroupGeometry = executionFacts.workgroupGeometry ?? record.workgroupGeometry;
      record.dispatchCount = executionFacts.dispatchCount ?? record.dispatchCount;
      record.captureArtifactIds = executionFacts.captureArtifactIds ?? record.captureArtifactIds;
    }

    record.timing = {
      wallMs: endTime - startTime,
      gpuMs: executionFacts?.gpuMs ?? null,
    };

    this._records.push(record);
    this._activeOp = null;
    return record;
  }

  emitRecord(stageName, options = {}) {
    if (!this._enabled) return null;

    const record = createOperatorExecutionRecord({
      ...options,
      stageName,
      modelHash: this._modelHash,
      adapterHash: this._adapterHash,
      runtimeConfigHash: this._runtimeConfigHash,
      executionPlanHash: this._executionPlanHash,
    });

    this._records.push(record);
    return record;
  }

  // ============================================================================
  // Timeline Access
  // ============================================================================

  getTimeline() {
    return this._records.slice();
  }

  getRecordsByLayer(layerIdx) {
    return this._records.filter((r) => r.layerIndex === layerIdx);
  }

  getRecordsByPhase(phase) {
    return this._records.filter((r) => r.phase === phase);
  }

  getRecordsByOpType(opType) {
    return this._records.filter((r) => r.opType === opType);
  }

  getRecordByOpId(opId) {
    return this._records.find((r) => r.opId === opId) ?? null;
  }

  get length() {
    return this._records.length;
  }

  clear() {
    this._records = [];
    this._activeOp = null;
  }

  toJSON() {
    return {
      modelHash: this._modelHash,
      adapterHash: this._adapterHash,
      runtimeConfigHash: this._runtimeConfigHash,
      executionPlanHash: this._executionPlanHash,
      recordCount: this._records.length,
      records: this._records,
    };
  }
}

// ============================================================================
// First-Divergence Detection
// ============================================================================

export function findFirstDivergence(baselineTimeline, observedTimeline, getDriftTolerance) {
  const len = Math.min(baselineTimeline.length, observedTimeline.length);

  for (let i = 0; i < len; i++) {
    const baseline = baselineTimeline[i];
    const observed = observedTimeline[i];

    if (baseline.opId !== observed.opId) {
      return {
        type: 'sequence_mismatch',
        index: i,
        baseline,
        observed,
        message: `Operator sequence diverged at index ${i}: ` +
          `expected "${baseline.opId}", got "${observed.opId}".`,
      };
    }

    if (getDriftTolerance && baseline.captureArtifactIds.length > 0) {
      const tolerance = getDriftTolerance(baseline.opType, baseline.dtype);
      if (tolerance !== null) {
        return {
          type: 'drift_check_needed',
          index: i,
          opId: baseline.opId,
          opType: baseline.opType,
          tolerance,
          baseline,
          observed,
        };
      }
    }
  }

  if (baselineTimeline.length !== observedTimeline.length) {
    return {
      type: 'length_mismatch',
      index: len,
      baselineLength: baselineTimeline.length,
      observedLength: observedTimeline.length,
      message: `Timeline length mismatch: baseline has ${baselineTimeline.length} records, ` +
        `observed has ${observedTimeline.length}.`,
    };
  }

  return null;
}
