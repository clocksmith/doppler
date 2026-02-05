export function resolveScheduleEnergy(schedule, taskCount) {
  if (!schedule) return Number.POSITIVE_INFINITY;
  if (schedule.violations > 0) return Number.POSITIVE_INFINITY;
  if (Number.isFinite(schedule.duplicates) && schedule.duplicates > 0) {
    return Number.POSITIVE_INFINITY;
  }
  if (Number.isFinite(schedule.missing) && schedule.missing > 0) {
    return Number.POSITIVE_INFINITY;
  }
  if (schedule.scheduled < taskCount) return Number.POSITIVE_INFINITY;
  return schedule.cycles;
}
