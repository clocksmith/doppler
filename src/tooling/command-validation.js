
import { isPlainObject } from '../formats/plain-object.js';

export function assertCommandRequestIsObject(commandRequest, surface) {
  if (!isPlainObject(commandRequest)) {
    throw new Error(
      `${surface} command: request must be a non-null plain object.`
    );
  }
  return commandRequest;
}

export function normalizeCommandOptions(options, surface) {
  if (options == null) return {};
  if (!isPlainObject(options)) {
    throw new Error(
      `${surface} command: options must be a plain object when provided.`
    );
  }
  return options;
}
