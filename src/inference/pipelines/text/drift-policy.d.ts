/**
 * Drift policy for operator-level differential debugging.
 *
 * Per-operator-class tolerances and propagation bound checking.
 */

import type { OperatorClass } from './stage-names.js';

export interface DriftTolerance {
  maxAbsDiff: number;
  maxRelDiff: number;
  propagationWeight: number;
}

export interface DriftCheckResult {
  withinBudget: boolean;
  reason: string;
  tolerance?: DriftTolerance;
  observed?: { maxAbsDiff: number; maxRelDiff: number };
}

export interface PropagationCheckResult {
  withinBound: boolean;
  reason: string;
  accumulated?: number;
  limit?: number;
  amplification?: number;
  threshold?: number;
}

export interface ObservedDrift {
  maxAbsDiff?: number;
  maxRelDiff?: number;
  propagationWeight?: number;
}

export declare function getDriftTolerance(
  operatorClass: OperatorClass | string | null,
  precisionMode: string | null
): DriftTolerance | null;

export declare function getDriftPolicyId(
  operatorClass: OperatorClass | string | null
): string | null;

export declare function getOperatorClasses(): string[];

export declare function checkDrift(
  operatorClass: OperatorClass | string | null,
  precisionMode: string | null,
  observed: ObservedDrift
): DriftCheckResult;

export declare function checkPropagationBound(
  precisionMode: string | null,
  layerDrifts: ObservedDrift[]
): PropagationCheckResult;
