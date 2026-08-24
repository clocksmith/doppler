export type MoEActiveExpertSelection = 'all' | 'topk-readback' | 'topk-route';

export interface ActiveExpertSchedule {
  selection: MoEActiveExpertSelection;
  activeExperts: number[];
  tokenCounts: Uint32Array | null;
}

export declare function resolveMoEActiveExpertSelection(
  selection: unknown
): MoEActiveExpertSelection;
export declare function buildActiveExpertScheduleFromIndices(
  indices: Uint32Array,
  numExperts: number,
  maxTokensPerExpert: number,
  selection?: MoEActiveExpertSelection
): ActiveExpertSchedule;

