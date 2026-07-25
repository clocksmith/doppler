export const COMMAND_CONTEXT_SCHEMA_VERSION = 1;

function freezeContext(context) {
  return Object.freeze(context);
}

export function createCommandContext(request) {
  if (!request || typeof request !== 'object') {
    throw new Error('tooling command context: normalized request must be an object.');
  }
  const command = typeof request.command === 'string' && request.command.trim()
    ? request.command.trim()
    : null;
  if (!command) {
    throw new Error('tooling command context: command is required.');
  }
  const workload = request.workload ?? null;
  const intent = request.intent ?? null;
  if (workload !== null && (typeof workload !== 'string' || workload.trim().length === 0)) {
    throw new Error('tooling command context: workload must be a non-empty string or null.');
  }
  if (intent !== null && (typeof intent !== 'string' || intent.trim().length === 0)) {
    throw new Error('tooling command context: intent must be a non-empty string or null.');
  }
  return freezeContext({
    schemaVersion: COMMAND_CONTEXT_SCHEMA_VERSION,
    command,
    workload,
    intent,
  });
}

export function assertCommandContextMatchesOptions(commandContext, options = {}) {
  if (!commandContext || commandContext.schemaVersion !== COMMAND_CONTEXT_SCHEMA_VERSION) {
    throw new Error(
      `tooling command context: schemaVersion must be ${COMMAND_CONTEXT_SCHEMA_VERSION}.`
    );
  }
  const comparisons = [
    ['command', options.command],
    ['workload', options.workload],
    ['intent', options.intent],
  ];
  for (const [field, mirror] of comparisons) {
    if (mirror !== undefined && mirror !== commandContext[field]) {
      throw new Error(
        `tooling command context: ${field} mirror "${mirror}" conflicts with normalized context ` +
        `"${commandContext[field]}".`
      );
    }
  }
  return commandContext;
}
