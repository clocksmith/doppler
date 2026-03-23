/**
 * Format a byte count as a human-readable string.
 * Guards against negative, NaN, and non-number inputs.
 *
 * @param {number} bytes
 * @returns {string}
 */
export function formatBytes(bytes) {
  if (typeof bytes !== 'number' || Number.isNaN(bytes)) return 'NaN';
  if (bytes < 0) return '0 B';

  const KB = 1024;
  const MB = KB * 1024;
  const GB = MB * 1024;

  if (bytes < KB) return `${bytes} B`;
  if (bytes < MB) return `${(bytes / KB).toFixed(1)} KB`;
  if (bytes < GB) return `${(bytes / MB).toFixed(1)} MB`;
  return `${(bytes / GB).toFixed(1)} GB`;
}
