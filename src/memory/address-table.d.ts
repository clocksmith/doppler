/**
 * Address Table
 * Agent-A | Domain: memory/
 *
 * Virtual address translation for segmented heap mode.
 * Encodes segment index + offset into a single 64-bit-safe number.
 *
 * Address format (fits in 53-bit JS safe integer):
 * - Upper 8 bits: segment index (0-255 segments)
 * - Lower 45 bits: offset within segment (up to 32TB per segment, but we use 4GB)
 *
 * @module memory/address-table
 */

/**
 * Decoded address parts
 */
export interface DecodedAddress {
  segmentIndex: number;
  offset: number;
}

/**
 * Address range chunk for split operations
 */
export interface AddressRangeChunk {
  segmentIndex: number;
  offset: number;
  length: number;
  virtualAddress: number;
}

/**
 * Address table constants
 */
export interface AddressTableConstants {
  SEGMENT_BITS: number;
  OFFSET_BITS: number;
  MAX_SEGMENTS: number;
  MAX_OFFSET: number;
}

/**
 * Virtual address table for segmented memory
 */
export declare class AddressTable {
  segmentSize: number;

  /**
   * @param segmentSize - Size of each segment in bytes
   */
  constructor(segmentSize: number);

  /**
   * Encode segment index and offset into virtual address
   * @param segmentIndex - Segment index (0-255)
   * @param offset - Byte offset within segment
   * @returns Virtual address
   */
  encode(segmentIndex: number, offset: number): number;

  /**
   * Decode virtual address into segment index and offset
   * @param virtualAddress - Virtual address to decode
   */
  decode(virtualAddress: number): DecodedAddress;

  /**
   * Get the segment index from a virtual address
   * @param virtualAddress - Virtual address
   */
  getSegmentIndex(virtualAddress: number): number;

  /**
   * Get the offset from a virtual address
   * @param virtualAddress - Virtual address
   */
  getOffset(virtualAddress: number): number;

  /**
   * Check if an address range spans multiple segments
   * @param virtualAddress - Starting virtual address
   * @param length - Length in bytes
   */
  spansSegments(virtualAddress: number, length: number): boolean;

  /**
   * Split an address range into per-segment chunks
   * Useful when a read/write spans segment boundaries
   * @param virtualAddress - Starting virtual address
   * @param length - Length in bytes
   */
  splitRange(virtualAddress: number, length: number): AddressRangeChunk[];

  /**
   * Calculate total virtual address space
   */
  getTotalAddressSpace(): number;
}

/**
 * Constants exported for other modules
 */
export declare const ADDRESS_TABLE_CONSTANTS: AddressTableConstants;
