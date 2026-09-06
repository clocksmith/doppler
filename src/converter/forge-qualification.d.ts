import type { TargetPlan } from '../config/target-plan.js';
import type { ModelIRV2 } from '../config/model-ir.js';

export declare function promoteQualifiedModelIRV2(modelIR: ModelIRV2, programBundle: Record<string, unknown>): ModelIRV2;

export declare function buildQualificationRecords(lowered: Record<string, unknown>): TargetPlan['qualification'];
