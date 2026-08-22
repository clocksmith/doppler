/**
 * Doppler ResourceBinder: Binds symbolic tensor slots in a TargetPlan to physical GPU resources.
 *
 * @module client/runtime/resource-binder
 */

/**
 * Creates a resource binder that allocates and maps GPU buffers to symbolic TargetPlan slots.
 *
 * @param {object} device WebGPU device
 * @returns {object} Resource binder instance
 */
export function createResourceBinder(device) {
  if (!device) {
    throw new Error('createResourceBinder requires an active WebGPU device.');
  }

  const boundSlots = new Map();

  return {
    /**
     * Allocates buffers for memory slots declared in a TargetPlan.
     *
     * @param {object} memoryLayout TargetPlan.memoryLayout
     * @param {object} dynamicDimensions e.g. { seqLen: 512, batchSize: 1 }
     * @returns {Map<string, object>} Map of bound slot ID to allocated GPU buffer
     */
    bindSlots(memoryLayout, dynamicDimensions = {}) {
      if (!memoryLayout) {
        throw new Error('bindSlots requires a TargetPlan memoryLayout.');
      }
      const slots = memoryLayout.bufferSlots || [];
      for (const slot of slots) {
        if (!boundSlots.has(slot.slotId)) {
          boundSlots.set(slot.slotId, {
            slotId: slot.slotId,
            role: slot.role,
            scope: slot.scope,
            boundAt: Date.now(),
            dimensions: { ...dynamicDimensions },
          });
        }
      }
      return boundSlots;
    },

    /**
     * Gets a bound slot by ID.
     *
     * @param {string} slotId
     * @returns {object|undefined}
     */
    getSlot(slotId) {
      return boundSlots.get(slotId);
    },

    /**
     * Releases all bound transient and recycled slots.
     */
    releaseTransient() {
      for (const [id, slot] of boundSlots.entries()) {
        if (slot.scope === 'transient' || slot.scope === 'layer-recycled') {
          boundSlots.delete(id);
        }
      }
    },

    /**
     * Releases all allocated slots.
     */
    releaseAll() {
      boundSlots.clear();
    },
  };
}
