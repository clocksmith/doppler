export function buildTaskIndex(tasks) {
  const byId = new Array(tasks.length);
  tasks.forEach((task) => {
    byId[task.id] = task;
  });
  return byId;
}
