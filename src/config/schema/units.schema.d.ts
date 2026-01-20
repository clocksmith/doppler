/**
 * Unit Constants (Invariants)
 *
 * Single source of truth for byte unit conversions.
 * These are mathematical constants, not tunables.
 */

/** 1 Kilobyte = 1024 bytes */
export declare const KB: number;

/** 1 Megabyte = 1024 * 1024 bytes */
export declare const MB: number;

/** 1 Gigabyte = 1024 * 1024 * 1024 bytes */
export declare const GB: number;

/**
 * Format bytes as human-readable string with units.
 * Example: 1536 -> "1.5 KB"
 */
export declare function formatBytes(bytes: number): string;

/**
 * Format bytes as compact string without spaces.
 * Example: 1536 -> "2KB"
 */
export declare function formatBytesCompact(bytes: number): string;
