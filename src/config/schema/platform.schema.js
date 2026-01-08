/**
 * Platform Config Schema Definitions
 *
 * Defines the structure for device-family platform configurations.
 * Platforms provide kernel overrides and tuning hints for specific GPU families.
 *
 * Note: GPU capabilities (hasSubgroups, hasF16, etc.) are detected at runtime,
 * not stored in config. Platforms assume those capabilities are available.
 *
 * @module config/schema/platform
 */

// This file only contains type definitions - no runtime exports
