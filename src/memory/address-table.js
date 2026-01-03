

// ============================================================================
// Constants
// ============================================================================

const SEGMENT_BITS = 8;
const OFFSET_BITS = 45;
const MAX_SEGMENTS = 1 << SEGMENT_BITS; // 256
// Note: Can't use << for OFFSET_BITS since JS bitwise ops are 32-bit. Use ** instead.
const MAX_OFFSET = 2 ** OFFSET_BITS - 1; // ~35TB (but we limit to segment size)

// ============================================================================
// Address Table Class
// ============================================================================


export class AddressTable {
  
  segmentSize;

  
  constructor(segmentSize) {
    this.segmentSize = segmentSize;

    // Validate segment size fits in offset bits
    if (segmentSize > MAX_OFFSET) {
      throw new Error(`Segment size ${segmentSize} exceeds max offset ${MAX_OFFSET}`);
    }
  }

  
  encode(segmentIndex, offset) {
    if (segmentIndex >= MAX_SEGMENTS) {
      throw new Error(`Segment index ${segmentIndex} exceeds max ${MAX_SEGMENTS - 1}`);
    }
    if (offset >= this.segmentSize) {
      throw new Error(`Offset ${offset} exceeds segment size ${this.segmentSize}`);
    }

    // Use BigInt for the shift to avoid precision loss, then convert back
    // Actually, since we're within 53 bits, we can use regular math
    return segmentIndex * (MAX_OFFSET + 1) + offset;
  }

  
  decode(virtualAddress) {
    const segmentIndex = Math.floor(virtualAddress / (MAX_OFFSET + 1));
    const offset = virtualAddress % (MAX_OFFSET + 1);

    return { segmentIndex, offset };
  }

  
  getSegmentIndex(virtualAddress) {
    return Math.floor(virtualAddress / (MAX_OFFSET + 1));
  }

  
  getOffset(virtualAddress) {
    return virtualAddress % (MAX_OFFSET + 1);
  }

  
  spansSegments(virtualAddress, length) {
    const startSegment = this.getSegmentIndex(virtualAddress);
    const endAddress = virtualAddress + length - 1;
    const endSegment = this.getSegmentIndex(endAddress);
    return startSegment !== endSegment;
  }

  
  splitRange(virtualAddress, length) {
    
    const chunks = [];
    let remaining = length;
    let currentAddress = virtualAddress;

    while (remaining > 0) {
      const { segmentIndex, offset } = this.decode(currentAddress);
      const availableInSegment = this.segmentSize - offset;
      const chunkLength = Math.min(remaining, availableInSegment);

      chunks.push({
        segmentIndex,
        offset,
        length: chunkLength,
        virtualAddress: currentAddress,
      });

      remaining -= chunkLength;
      currentAddress += chunkLength;
    }

    return chunks;
  }

  
  getTotalAddressSpace() {
    return MAX_SEGMENTS * this.segmentSize;
  }
}

// ============================================================================
// Exports
// ============================================================================


export const ADDRESS_TABLE_CONSTANTS = {
  SEGMENT_BITS,
  OFFSET_BITS,
  MAX_SEGMENTS,
  MAX_OFFSET,
};
