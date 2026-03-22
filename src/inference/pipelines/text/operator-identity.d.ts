/**
 * Canonical operator identity for operator-level differential debugging.
 *
 * Builds deterministic opIds from (section, layerIdx, stageName).
 */

import type { StageName, OperatorClass } from './stage-names.js';

export interface OperatorMeta {
  opId: string;
  stageName: StageName;
  operatorClass: OperatorClass | null;
  phase: 'prefill' | 'decode' | 'both' | null;
  layerIdx: number | null;
  tokenIndex: number | null;
  dtype: string | null;
  quantizationMode: string | null;
  shapeSignature: string | null;
}

export interface SequencedOperatorMeta extends OperatorMeta {
  sequenceIndex: number;
}

export declare function buildOpId(stageName: StageName, layerIdx?: number | null): string;

export declare function buildOpIdFromProbeStage(
  probeStageName: string,
  layerIdx?: number | null
): string;

export declare function buildOpIdFromExecutionStep(resolvedStep: {
  section: string;
  op: string;
  layers?: number[];
}): string;

export declare function buildOperatorMeta(
  stageName: StageName,
  options: {
    layerIdx?: number | null;
    phase?: 'prefill' | 'decode' | 'both' | null;
    tokenIndex?: number | null;
    dtype?: string | null;
    quantizationMode?: string | null;
    shapeSignature?: string | null;
  }
): OperatorMeta;

export declare class OperatorSequence {
  constructor();
  record(opMeta: OperatorMeta): SequencedOperatorMeta;
  get length(): number;
  getOps(): SequencedOperatorMeta[];
  getOpById(opId: string): SequencedOperatorMeta | null;
  getOpsByLayer(layerIdx: number): SequencedOperatorMeta[];
  getOpsByClass(operatorClass: OperatorClass): SequencedOperatorMeta[];
  clear(): void;
  toJSON(): SequencedOperatorMeta[];
}
