/**
 * Validate that the command request is a non-null plain object.
 * Throws a descriptive error if not.
 *
 * @param {*} commandRequest - The raw command request to validate.
 * @param {string} surface - The surface name for error context ('node' | 'browser').
 * @returns {object} The validated command request (same reference).
 */
export function assertCommandRequestIsObject(commandRequest: any, surface: string): object;
/**
 * Validate that options, when provided, is a plain object.
 *
 * @param {*} options - The options value to validate.
 * @param {string} surface - The surface name for error context.
 * @returns {object} The validated options, or an empty object if nullish.
 */
export function normalizeCommandOptions(options: any, surface: string): object;
