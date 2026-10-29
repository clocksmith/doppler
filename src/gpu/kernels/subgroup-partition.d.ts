/** Refuses a partition that can cross a physical subgroup, including unknown ranges. */
export function assertSubgroupPartition(lanes: number, info: {
  subgroupMinSize?: number | null; subgroupMaxSize?: number | null;
} | null | undefined): void;
