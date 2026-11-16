// =============================================================================
// Unit Constants (Invariants)
// =============================================================================
// These are mathematical constants, not tunables.
// Single source of truth for byte unit conversions.

export const KB = 1024;
export const MB = 1024 * 1024;
export const GB = 1024 * 1024 * 1024;

// =============================================================================
// Format Helpers
// =============================================================================

export function formatBytes(bytes) {
  if (typeof bytes !== 'number' || Number.isNaN(bytes)) return 'NaN';
  if (bytes < 0) return '0 B';
  if (bytes < KB) return `${bytes} B`;
  if (bytes < MB) return `${(bytes / KB).toFixed(1)} KB`;
  if (bytes < GB) return `${(bytes / MB).toFixed(1)} MB`;
  return `${(bytes / GB).toFixed(1)} GB`;
}

export function formatBytesCompact(bytes) {
  if (typeof bytes !== 'number' || Number.isNaN(bytes)) return 'NaN';
  if (bytes < 0) return '0B';
  if (bytes < MB) return `${(bytes / KB).toFixed(0)}KB`;
  if (bytes < GB) return `${(bytes / MB).toFixed(0)}MB`;
  return `${(bytes / GB).toFixed(1)}GB`;
}
