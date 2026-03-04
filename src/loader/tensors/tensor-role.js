export function getTensorNamesByRole(tensorLocations, role, group = null) {
  if (!tensorLocations) return [];

  const names = [];
  for (const [name, location] of tensorLocations) {
    if (!location || location.role !== role) continue;
    if (group != null && location.group !== group) continue;
    names.push(name);
  }

  return names.sort((a, b) => a.localeCompare(b));
}
