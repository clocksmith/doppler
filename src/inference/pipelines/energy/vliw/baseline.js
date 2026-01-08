export function buildBaselinePriorities(tasks) {
  const priorities = new Float32Array(tasks.length);
  const bundleCounts = new Map();
  let maxBundle = 0;
  tasks.forEach((task) => {
    const bundle = Number.isFinite(task.bundle) ? task.bundle : 0;
    if (bundle > maxBundle) maxBundle = bundle;
    bundleCounts.set(bundle, (bundleCounts.get(bundle) || 0) + 1);
  });
  let maxCount = 1;
  bundleCounts.forEach((count) => {
    if (count > maxCount) maxCount = count;
  });
  const scale = maxCount + 1;
  const bundleOffsets = new Map();
  tasks.forEach((task) => {
    const bundle = Number.isFinite(task.bundle) ? task.bundle : 0;
    const pos = bundleOffsets.get(bundle) || 0;
    bundleOffsets.set(bundle, pos + 1);
    const rank = bundle * scale + pos;
    priorities[task.id] = -rank;
  });
  return priorities;
}
