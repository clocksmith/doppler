/**
 * Shared command validation helpers used by both node-command-runner.js and
 * browser-command-runner.js.
 *
 * This module exists to prevent validation logic from diverging between the
 * two runner surfaces. Any command-level validation that must behave
 * identically on both surfaces should live here.
 */

import { isPlainObject } from '../utils/plain-object.js';

/**
 * Validate that the command request is a non-null plain object.
 * Throws a descriptive error if not.
 *
 * @param {*} commandRequest - The raw command request to validate.
 * @param {string} surface - The surface name for error context ('node' | 'browser').
 * @returns {object} The validated command request (same reference).
 */
export function assertCommandRequestIsObject(commandRequest, surface) {
  if (!isPlainObject(commandRequest)) {
    throw new Error(
      `${surface} command: request must be a non-null plain object.`
    );
  }
  return commandRequest;
}

/**
 * Validate that options, when provided, is a plain object.
 *
 * @param {*} options - The options value to validate.
 * @param {string} surface - The surface name for error context.
 * @returns {object} The validated options, or an empty object if nullish.
 */
export function normalizeCommandOptions(options, surface) {
  if (options == null) return {};
  if (!isPlainObject(options)) {
    throw new Error(
      `${surface} command: options must be a plain object when provided.`
    );
  }
  return options;
}
